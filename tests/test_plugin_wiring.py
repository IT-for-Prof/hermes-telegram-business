"""Wiring tests: register(ctx) queues a factory; the factory adds handlers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import telegram_business_plugin as plugin


class _FakeCtx:
    def __init__(self):
        self.factories = []
        self.llm = MagicMock()
        self.aux_tasks = {}

    def register_auxiliary_task(self, key, **kwargs):
        self.aux_tasks[key] = kwargs

    def register_tool(self, **kwargs):
        self.tools = getattr(self, "tools", {})
        self.tools[kwargs["name"]] = kwargs

    def register_telegram_handler(self, factory):
        assert callable(factory)
        self.factories.append(factory)


def test_register_queues_one_factory():
    ctx = _FakeCtx()
    plugin.register(ctx)
    assert len(ctx.factories) == 1


def test_drafts_and_gists_use_own_tasks_without_reasoning_and_with_a_cost_tag():
    ctx = _FakeCtx()
    plugin.register(ctx)
    for task, tag in ((plugin.DRAFT_TASK, "telegram-business-draft"), (plugin.GIST_TASK, "telegram-business-gist")):
        assert ctx.aux_tasks[task]["defaults"] == {"extra_body": {
            "reasoning_effort": "none", "metadata": {"generation_name": tag, "tags": [tag]}}}


def _wired(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = _FakeCtx()
    plugin.register(ctx)
    application = MagicMock()
    ctx.factories[0](application, SimpleNamespace(_bot=MagicMock(), name="telegram"))
    return ctx, application


def _handler(application, callback_name):
    """The callback registered under that name in the latest wiring, and how many handlers sit in group -1."""
    from telegram.ext import MessageHandler
    callbacks = [c.args[-1] for c in MessageHandler.call_args_list if getattr(c.args[-1], "__name__", "") == callback_name]
    assert callbacks, callback_name
    early = sum(kw.get("group") == -1 for _, kw in application.add_handler.call_args_list)
    return callbacks[-1], early


def _manager(application):
    import inspect
    return inspect.getclosurevars(_handler(application, "_on_owner_voice")[0]).nonlocals["manager"]


@pytest.mark.asyncio
async def test_gist_wording_uses_the_gist_task_and_rules(tmp_path, monkeypatch):
    from telegram_business_plugin.manager import GIST_OUTPUT_RULES
    ctx, application = _wired(tmp_path, monkeypatch)
    ctx.llm.complete.return_value = SimpleNamespace(text="Да, в пятницу.")
    gen = _manager(application)._draft_generator
    assert await gen("prompt", "200", rules=GIST_OUTPUT_RULES) == "Да, в пятницу."
    messages = ctx.llm.complete.call_args.args[0]
    assert ctx.llm.complete.call_args.kwargs["task"] == plugin.GIST_TASK
    assert messages[0]["content"].endswith(GIST_OUTPUT_RULES)
    await gen("prompt", "200")
    assert ctx.llm.complete.call_args.kwargs["task"] == plugin.DRAFT_TASK


@pytest.mark.asyncio
async def test_voice_note_is_transcribed_only_while_a_gist_waits(tmp_path, monkeypatch):
    import sys
    import types

    from telegram.ext import ApplicationHandlerStop
    heard = []
    stt = types.ModuleType("tools.transcription_tools")
    stt.transcribe_audio = lambda path: heard.append(path) or {"success": True, "transcript": "да в пятницу"}
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", stt)
    ctx, application = _wired(tmp_path, monkeypatch)
    on_voice, early = _handler(application, "_on_owner_voice")
    assert early == 2  # text and voice capture both run before Hermes's own handlers
    downloads = []

    async def _download(path):
        downloads.append(path)
        Path(path).write_bytes(b"ogg")

    tg_file = SimpleNamespace(download_to_drive=_download)

    async def _get_file():
        return tg_file

    update = SimpleNamespace(message=SimpleNamespace(chat=SimpleNamespace(id=100), voice=SimpleNamespace(get_file=_get_file)))
    await on_voice(update, None)  # nothing waits: left to Hermes, nothing downloaded
    assert downloads == [] and heard == []

    manager = _manager(application)
    taken = []

    async def _take(owner_chat_id, text):
        taken.append((owner_chat_id, text))
        return True

    manager._gist_capture["100"] = {"conn_id": "c", "chat_id": "200", "origin": "gist_card", "card_id": None}
    monkeypatch.setattr(manager, "_take_gist", _take)
    with pytest.raises(ApplicationHandlerStop):
        await on_voice(update, None)
    assert taken == [("100", "да в пятницу")] and heard == downloads
    assert not Path(downloads[0]).exists()  # the downloaded clip is removed


@pytest.mark.asyncio
async def test_failed_transcription_reaches_the_owner(tmp_path, monkeypatch):
    import sys
    import types
    stt = types.ModuleType("tools.transcription_tools")
    stt.transcribe_audio = lambda path: {"success": False, "error": "no backend"}
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", stt)
    ctx, application = _wired(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="no backend"):
        await plugin._transcribe_voice(SimpleNamespace(voice=SimpleNamespace(
            get_file=lambda: asyncio.sleep(0, result=SimpleNamespace(
                download_to_drive=lambda path: asyncio.sleep(0, result=Path(path).write_bytes(b"x")))))))


def test_factory_wires_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = _FakeCtx()
    plugin.register(ctx)
    factory = ctx.factories[0]

    application = MagicMock()
    adapter = SimpleNamespace(_bot=MagicMock(), name="telegram")

    factory(application, adapter)

    # 7 handlers: BusinessConnection, 2x business message, bd: callback,
    # /biz command, owner-DM edit/gist capture, owner-DM voice gist.
    assert application.add_handler.call_count == 7

    # The edit-capture handler must be registered in group -1 so it runs
    # before (and can yield to) the core adapter's text handler.
    groups = [
        kwargs.get("group")
        for args, kwargs in application.add_handler.call_args_list
    ]
    assert -1 in groups

    # Plugin state DB created under HERMES_HOME, not the core state.db.
    assert (tmp_path / "telegram-business" / "state.db").exists()


# --- Telegram reconnect: the gateway rebuilds the adapter and re-runs the factory -------------

import asyncio  # noqa: E402

import pytest  # noqa: E402


def _bot():
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=lambda **kw: asyncio.sleep(0, result=kw))
    return bot


@pytest.mark.asyncio
async def test_send_uses_the_adapter_of_the_latest_wiring():
    old, new = SimpleNamespace(_bot=_bot()), SimpleNamespace(_bot=_bot())
    state = {"adapter": old}
    old._bot = None                 # the old adapter disconnected...
    state["adapter"] = new          # ...and the factory ran again with the new one
    await plugin._send_via(state, chat_id=1, text="draft")
    new._bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_waits_until_the_new_adapter_is_connected(monkeypatch):
    monkeypatch.setattr(plugin, "_BOT_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(plugin, "_BOT_POLL_SECONDS", 0.05)
    adapter = SimpleNamespace(_bot=_bot(), is_connected=False)   # bot exists, still connecting
    state = {"adapter": adapter}

    async def connected():
        await asyncio.sleep(0.2)
        assert adapter._bot.send_message.call_count == 0
        adapter.is_connected = True

    asyncio.get_running_loop().create_task(connected())
    await plugin._send_via(state, chat_id=1, text="draft")
    adapter._bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_waits_for_the_bot_during_a_reconnect(monkeypatch):
    monkeypatch.setattr(plugin, "_BOT_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(plugin, "_BOT_POLL_SECONDS", 0.05)
    adapter = SimpleNamespace(_bot=None)
    state = {"adapter": adapter}

    async def reconnect():
        await asyncio.sleep(0.2)
        adapter._bot = _bot()

    asyncio.get_running_loop().create_task(reconnect())
    await plugin._send_via(state, chat_id=1, text="draft")
    adapter._bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_fails_loudly_when_no_bot_comes_back(monkeypatch):
    monkeypatch.setattr(plugin, "_BOT_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(plugin, "_BOT_POLL_SECONDS", 0.05)
    state = {"adapter": SimpleNamespace(_bot=None)}
    with pytest.raises(RuntimeError, match="not connected"):
        await plugin._send_via(state, chat_id=1, text="draft")


@pytest.mark.asyncio
async def test_after_rewiring_messages_go_through_the_new_adapter(tmp_path, monkeypatch):
    """Through register(): connect, rebuild (factory runs again), then the onboarding DM must use the new bot."""
    import telegram.ext as ext

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = _FakeCtx()
    plugin.register(ctx)
    factory = ctx.factories[0]
    first, second = SimpleNamespace(_bot=_bot()), SimpleNamespace(_bot=_bot())
    factory(MagicMock(), first)
    first._bot = None                     # old adapter disconnected by the gateway
    factory(MagicMock(), second)
    on_connection = ext.BusinessConnectionHandler.call_args_list[-1].args[0]
    connection = SimpleNamespace(id="c1", user=SimpleNamespace(id=42, full_name="Owner"),
                                 user_chat_id=100, is_enabled=True, rights=SimpleNamespace(can_reply=True))
    await on_connection(SimpleNamespace(business_connection=connection), None)
    second._bot.send_message.assert_called()


@pytest.mark.asyncio
async def test_secretary_reply_tool_runs_on_the_gateway_loop(tmp_path, monkeypatch):
    import json
    ctx = _FakeCtx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.register(ctx)
    tool = ctx.tools["secretary_reply"]
    assert tool["toolset"] == "telegram-business"
    assert tool["schema"]["parameters"]["required"] == ["contact", "gist"]
    assert tool["check_fn"]() is False  # nothing to reply through before the gateway wires the plugin
    application = MagicMock()
    ctx.factories[0](application, SimpleNamespace(_bot=MagicMock(), name="telegram"))
    assert tool["check_fn"]() is True
    manager = _manager(application)
    seen = []

    async def _reply(*, contact, gist):
        seen.append((contact, gist, asyncio.get_running_loop()))
        return {"status": "shown", "contact": "Carol"}

    monkeypatch.setattr(manager, "reply_from_dm", _reply)
    result = await asyncio.to_thread(tool["handler"], {"contact": "Carol", "gist": "да"})  # Hermes's worker thread
    assert json.loads(result) == {"status": "shown", "contact": "Carol"}
    assert seen == [("Carol", "да", asyncio.get_running_loop())]
