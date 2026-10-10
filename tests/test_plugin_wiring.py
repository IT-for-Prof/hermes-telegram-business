"""Wiring tests: register(ctx) queues a factory; the factory adds handlers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import telegram_business_plugin as plugin


class _FakeCtx:
    def __init__(self):
        self.factories = []
        self.llm = MagicMock()

    def register_telegram_handler(self, factory):
        assert callable(factory)
        self.factories.append(factory)


def test_register_queues_one_factory():
    ctx = _FakeCtx()
    plugin.register(ctx)
    assert len(ctx.factories) == 1


def test_factory_wires_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = _FakeCtx()
    plugin.register(ctx)
    factory = ctx.factories[0]

    application = MagicMock()
    adapter = SimpleNamespace(_bot=MagicMock(), name="telegram")

    factory(application, adapter)

    # 6 handlers: BusinessConnection, 2x business message, bd: callback,
    # /biz command, owner-DM edit capture.
    assert application.add_handler.call_count == 6

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
