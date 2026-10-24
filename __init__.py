"""hermes-telegram-business — observe-with-approval Telegram Business Mode.

Turns a Hermes Telegram bot into a secretary for a Telegram Business
account: customers message the owner, the plugin drafts a reply via the
host LLM, and the OWNER approves every message before it goes out: each
card offers 2-3 numbered replies with buttons 1-3, plus [✍ Своя суть]
(the owner's gist by text or voice, worded into a reply he then sends),
[✎ Edit] and [✕ Discard]. There is no auto-send.

Wiring (all through the public Hermes plugin surface — zero core edits):

* ``ctx.register_telegram_handler`` — PTB handlers for BusinessConnection
  updates, business_message / edited_business_message updates, a
  pattern-scoped ``bd:`` CallbackQueryHandler for the draft buttons,
  owner-DM text and voice handlers for Edit and gist capture, and a
  ``/biz`` CommandHandler.
* ``ctx.register_tool`` — ``secretary_reply``: the Hermes agent hands over the
  owner's gist from the bot DM ("Ивану: да, в пятницу", typed or spoken).
* ``ctx.llm`` — host-owned drafting (user's active model + auth).
* Plugin-owned SQLite state at ``<HERMES_HOME>/telegram-business/state.db``.

Requires Hermes Agent with ``register_telegram_handler`` support
(hermes-agent PR #59159) and python-telegram-bot >= 21.1 (ships with the
Telegram platform plugin).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

try:  # Normal package import (Hermes plugin loader, tests via conftest alias)
    from .manager import BusinessModeManager, CALLBACK_PREFIX, DRAFT_OUTPUT_RULES, GIST_OUTPUT_RULES
    from .state import BusinessStateDB
    from .memory import load as load_contact_memory
except ImportError:  # pragma: no cover - loaded as a bare module (no package)
    import importlib.util as _ilu
    import sys as _sys

    def _load(_name: str):
        _path = Path(__file__).resolve().parent / f"{_name}.py"
        _spec = _ilu.spec_from_file_location(f"telegram_business_{_name}", _path)
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod

    _manager_mod = _load("manager")
    _state_mod = _load("state")
    BusinessModeManager = _manager_mod.BusinessModeManager
    CALLBACK_PREFIX = _manager_mod.CALLBACK_PREFIX
    DRAFT_OUTPUT_RULES = _manager_mod.DRAFT_OUTPUT_RULES
    GIST_OUTPUT_RULES = _manager_mod.GIST_OUTPUT_RULES
    BusinessStateDB = _state_mod.BusinessStateDB
    load_contact_memory = _load("memory").load

logger = logging.getLogger(__name__)

_DEFAULT_PERSONA = (
    "You are drafting a short, friendly reply on behalf of the account "
    "owner. Match the tone of a personal message — warm, direct, and "
    "concise. Do not introduce yourself as an assistant or AI. Reply in "
    "the same language the customer used."
)


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home)
    return Path.home() / ".hermes"


# The gateway rebuilds the Telegram adapter after a network error and re-runs our factory with the
# new one; sends must always go through the latest adapter, never one captured at first wiring.
_BOT_WAIT_SECONDS = 60.0  # how long a send waits for the bot to come back during a reconnect
_BOT_POLL_SECONDS = 1.0


async def _send_via(state: dict, **kwargs):
    kwargs.setdefault("parse_mode", None)
    kwargs.setdefault("disable_web_page_preview", True)
    return await _bot_call(state, "send_message", **kwargs)


async def _bot_call(state: dict, method: str, **kwargs):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _BOT_WAIT_SECONDS
    while True:
        adapter = state.get("adapter")
        bot = getattr(adapter, "_bot", None)
        # A rebuilt adapter gets its bot before it finishes connecting; wait for is_connected too.
        if bot is not None and getattr(adapter, "is_connected", True):
            return await getattr(bot, method)(**kwargs)
        if loop.time() >= deadline:
            raise RuntimeError("Telegram bot is not connected")
        await asyncio.sleep(_BOT_POLL_SECONDS)


async def _transcribe_voice(message: Any) -> str:
    """The owner's voice note as text, through Hermes's own speech-to-text (local whisper here)."""
    from tools.transcription_tools import transcribe_audio

    tg_file = await message.voice.get_file()
    with tempfile.TemporaryDirectory(prefix="tb-gist-") as tmp:
        path = str(Path(tmp) / "gist.ogg")
        await tg_file.download_to_drive(path)
        result = await asyncio.get_running_loop().run_in_executor(None, transcribe_audio, path)
    if not result.get("success") or not (result.get("transcript") or "").strip():
        raise RuntimeError(result.get("error") or "empty transcript")
    return result["transcript"].strip()


_CONFIG_KEYS = ("debounce_seconds", "draft_ttl_hours", "max_customer_text_chars", "owner_persona",
                "allowed_owner_ids")


def _plugin_config(ctx: Any) -> dict:
    """This plugin's settings, per key (best-effort).

    Preferred layout (read through the host API ``ctx.get_config``)::

        plugins:
          entries:
            telegram-business:
              settings:
                debounce_seconds: 8
                draft_ttl_hours: 24
                max_customer_text_chars: 4000
                owner_persona: "..."
                allowed_owner_ids: [123456789]   # Telegram user ids whose Business accounts may connect

    The legacy flat layout (keys directly under ``telegram-business``) still
    works; ``settings`` wins per key. Never parse config.yaml ourselves: PyYAML is
    not importable inside the gateway, and a silent failure here used to swap in
    the built-in English persona.
    """
    cfg: dict = {}
    try:
        from hermes_cli.config import load_config_readonly

        entry = (((load_config_readonly() or {}).get("plugins") or {}).get("entries") or {}).get(
            "telegram-business") or {}
        if isinstance(entry, dict):
            cfg = {k: entry[k] for k in _CONFIG_KEYS if entry.get(k) is not None}
    except Exception as exc:
        logger.warning("telegram-business: legacy config read failed: %s", exc)
    getter = getattr(ctx, "get_config", None)
    if callable(getter):
        for key in _CONFIG_KEYS:
            try:
                value = getter(key)
            except Exception as exc:
                logger.warning("telegram-business: ctx.get_config(%r) failed: %s", key, exc)
                continue
            if value is not None:
                cfg[key] = value
    return cfg


DRAFT_TASK = "telegram_business_draft"
GIST_TASK = "telegram_business_gist"


SECRETARY_TOOL_SCHEMA = {
    "name": "secretary_reply",
    "description": (
        "Prepare a reply to one of the owner's Telegram Business contacts from the owner's gist and show it to "
        "him in this chat with Send / Edit / Discard (or for copying, when Telegram no longer lets the bot reply "
        "in that chat); nothing reaches the contact until he taps Send. Call it when "
        "the owner tells you what to answer a contact, e.g. 'Ивану: да, в пятницу', typed or in a voice note. "
        "Pass the owner's own words for the reply as `gist`, verbatim: do not reword, translate, or add dates or "
        "facts. If the result lists several candidates or none, ask the owner which chat he means and call again "
        "with the chat id. After 'shown', confirm in a few words; do not repeat the reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "contact": {"type": "string", "description": "The contact's name in its base form, as it would appear "
                        "in the chat ('Иван' for 'Ивану'), or a chat id."},
            "gist": {"type": "string", "description": "The owner's words for the reply, exactly as he said them."},
        },
        "required": ["contact", "gist"],
    },
}
_TOOL_TIMEOUT_S = 120.0  # wording the gist is one model call; the preview send may wait for a reconnecting bot


def _owner_ids(value: Any) -> Optional[frozenset]:
    """``allowed_owner_ids`` as a set of id strings: a list, one number or a comma-separated string; None if unset."""
    if value is None or value == "":
        return None
    items = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    return frozenset(str(i).strip() for i in items if str(i).strip())


def _task_defaults(tag: str) -> dict:
    # Reasoning off: Qwen thinks by default and spent ~1.8k output tokens per draft; the replay eval showed the
    # same style, commas and ё without it (2026-10-23). The tag lets Langfuse sum each task's spend.
    return {"extra_body": {"reasoning_effort": "none", "metadata": {"generation_name": tag, "tags": [tag]}}}


def register(ctx: Any) -> None:
    cfg = _plugin_config(ctx)
    # Own auxiliary slots (see _task_defaults); reasoning goes back on with
    # auxiliary.<task>.extra_body in config.yaml.
    ctx.register_auxiliary_task(DRAFT_TASK, display_name="Telegram Business drafts",
                                description="Reply drafts for Telegram Business chats",
                                defaults=_task_defaults("telegram-business-draft"))
    ctx.register_auxiliary_task(GIST_TASK, display_name="Telegram Business gist replies",
                                description="The owner's gist worded into his reply",
                                defaults=_task_defaults("telegram-business-gist"))
    debounce = float(cfg.get("debounce_seconds", 8.0))
    ttl_hours = float(cfg.get("draft_ttl_hours", 24.0))
    max_chars = int(cfg.get("max_customer_text_chars", 4000))
    custom_persona = cfg.get("owner_persona")
    persona = str(custom_persona or _DEFAULT_PERSONA)
    persona_source = f"{'config' if custom_persona else 'BUILT-IN DEFAULT'}, {len(persona)} chars"
    logger.info("telegram-business: persona from %s", persona_source)
    allowed_owner_ids = _owner_ids(cfg.get("allowed_owner_ids"))
    if allowed_owner_ids is None:
        logger.warning("telegram-business: allowed_owner_ids not set, any Business account may connect the bot")

    # Deferred singletons — constructed on first connect so import stays light.
    _state: dict = {"db": None, "manager": None, "adapter": None, "loop": None}

    def _secretary_reply(args: dict) -> str:
        manager, loop = _state["manager"], _state["loop"]
        if manager is None or loop is None:
            return json.dumps({"status": "failed", "error": "the Telegram secretary is not connected"})
        # Only the account owner, from his own Telegram chat with the bot, and only among his own chats: anyone
        # can connect a Business account to the bot, and cron or CLI turns have no owner to ask.
        from gateway.session_context import get_session_env
        user = get_session_env("HERMES_SESSION_USER_ID")
        if get_session_env("HERMES_SESSION_PLATFORM") != "telegram" or not user:
            return json.dumps({"status": "failed", "error": "only from the owner's Telegram chat with the bot"})
        # Hermes runs tool handlers in a worker thread with its own loop; the bot's sends must run on the gateway
        # loop the adapter lives on, or they deadlock (tools/send_message_tool.py, _dispatch_on_gateway_loop).
        future = asyncio.run_coroutine_threadsafe(manager.reply_from_dm(
            owner_user_id=user, contact=str(args.get("contact", "")), gist=str(args.get("gist", ""))), loop)
        try:
            return json.dumps(future.result(timeout=_TOOL_TIMEOUT_S), ensure_ascii=False)
        except TimeoutError:
            future.cancel()  # a late preview would contradict the failure reported to the agent
            return json.dumps({"status": "failed", "error": "timed out; nothing was shown"})

    ctx.register_tool(name="secretary_reply", toolset="telegram-business", schema=SECRETARY_TOOL_SCHEMA,
                      handler=_secretary_reply, check_fn=lambda: _state["manager"] is not None,
                      description="Reply to a Telegram Business contact from the owner's gist", emoji="✍")

    def _get_manager() -> BusinessModeManager:
        if _state["manager"] is not None:
            return _state["manager"]

        db = BusinessStateDB(_hermes_home() / "telegram-business" / "state.db")
        _state["db"] = db
        try:
            memory = load_contact_memory(_hermes_home())
        except Exception as exc:
            logger.warning("telegram-business: contact memory disabled: %s", exc)
            memory = None
        logger.info("telegram-business: contact memory %s", "on (Honcho)" if memory else "off")

        async def _send(**kwargs):
            return await _send_via(_state, **kwargs)

        async def _draft(prompt_text: str, customer_chat_id: str, rules: str = DRAFT_OUTPUT_RULES) -> str:
            messages = [
                {"role": "system", "content": f"{persona}\n\n{rules}"},
                {"role": "user", "content": prompt_text},
            ]
            task = {DRAFT_OUTPUT_RULES: DRAFT_TASK, GIST_OUTPUT_RULES: GIST_TASK}[rules]
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: ctx.llm.complete(
                    messages,
                    max_tokens=500,
                    timeout=60.0,
                    purpose="telegram-business draft",
                    task=task,
                ),
            )
            return (result.text or "").strip()

        _state["manager"] = BusinessModeManager(
            session_db=db,
            send_message=_send,
            draft_generator=_draft,
            debounce_seconds=debounce,
            draft_ttl_hours=ttl_hours,
            max_customer_text_chars=max_chars,
            persona_source=persona_source,
            memory=memory,
            edit_markup=lambda **kw: _bot_call(_state, "edit_message_reply_markup", **kw),
            allowed_owner_ids=allowed_owner_ids,
        )
        return _state["manager"]

    # ------------------------------------------------------------------
    # The handler factory the Telegram adapter invokes at connect() time.
    # PTB imports live inside so register() works without PTB installed.
    # ------------------------------------------------------------------

    def _wire(application: Any, adapter: Any) -> None:
        from telegram.ext import (
            BusinessConnectionHandler,
            CallbackQueryHandler,
            CommandHandler,
            MessageHandler,
            filters,
        )

        _state["adapter"] = adapter  # latest wiring wins (see _send_via)
        try:  # connect() calls us on the gateway loop; a later rewire may not, and keeps the loop we have
            _state["loop"] = asyncio.get_running_loop()
        except RuntimeError:
            pass
        manager = _get_manager()

        async def _on_business_connection(update, context):
            conn = getattr(update, "business_connection", None)
            if conn is None:
                return
            try:
                await manager.handle_connection_update(conn)
            except Exception as exc:
                logger.exception("telegram-business: connection handler failed: %s", exc)

        async def _on_business_message(update, context):
            message = getattr(update, "business_message", None)
            edited = message is None
            if edited:
                message = getattr(update, "edited_business_message", None)
            if message is None:
                return
            try:
                await manager.handle_business_message(message, edited=edited)
            except Exception as exc:
                logger.exception("telegram-business: message handler failed: %s", exc)

        async def _on_draft_button(update, context):
            query = update.callback_query
            if not query or not query.data:
                return
            caller_id = str(getattr(query.from_user, "id", "")) or None
            try:
                await manager.handle_callback(
                    data=query.data,
                    caller_user_id=caller_id,
                    answer=query.answer,
                    edit_message_text=query.edit_message_text,
                )
            except Exception as exc:
                logger.exception("telegram-business: draft callback failed: %s", exc)
                try:
                    await query.answer(text="⚠️ Action failed.")
                except Exception:
                    pass

        async def _on_owner_dm(update, context):
            """Edit-capture: the owner's next DM after tapping Edit.

            Registered in group -1 (runs before the default group where
            the core adapter's handlers live). When the message is
            consumed as an edit override we raise ApplicationHandlerStop
            so the core text pipeline never sees it; otherwise we return
            normally and PTB continues into group 0 unchanged.
            """
            from telegram.ext import ApplicationHandlerStop

            message = getattr(update, "message", None)
            chat = getattr(message, "chat", None)
            if message is None or chat is None:
                return
            try:
                consumed = await manager.maybe_handle_edit_capture(
                    owner_chat_id=str(chat.id),
                    text=getattr(message, "text", "") or "",
                )
            except Exception as exc:
                logger.exception("telegram-business: edit capture failed: %s", exc)
                consumed = False
            if consumed:
                raise ApplicationHandlerStop

        async def _on_owner_voice(update, context):
            """A voice note in the bot DM: the awaited gist (then Hermes never sees it), else left to Hermes."""
            from telegram.ext import ApplicationHandlerStop

            message = getattr(update, "message", None)
            chat = getattr(message, "chat", None)
            if message is None or chat is None:
                return
            try:
                consumed = await manager.handle_owner_voice(
                    owner_chat_id=str(chat.id), transcribe=lambda: _transcribe_voice(message))
            except Exception as exc:
                logger.exception("telegram-business: voice gist failed: %s", exc)
                consumed = False
            if consumed:
                raise ApplicationHandlerStop

        async def _on_biz_command(update, context):
            message = getattr(update, "message", None)
            if message is None:
                return
            user = getattr(message, "from_user", None)
            chat = getattr(message, "chat", None)
            text = (getattr(message, "text", "") or "").strip()
            args = text.split()[1:]
            try:
                reply = await manager.handle_biz_command(
                    owner_user_id=str(getattr(user, "id", "")),
                    owner_chat_id=str(getattr(chat, "id", "")),
                    args=args,
                )
            except Exception as exc:
                logger.exception("telegram-business: /biz failed: %s", exc)
                reply = f"⚠️ /biz failed: {exc}"
            try:
                if chat is not None:
                    await _send_via(_state, chat_id=chat.id, text=reply)
            except Exception:
                logger.debug("telegram-business: /biz reply send failed", exc_info=True)

        # Connection lifecycle (established / edited / ended).
        application.add_handler(BusinessConnectionHandler(_on_business_connection))
        # Every Business message, both sides: media without text still goes into
        # the chat log as context ("[photo]"); only the other party's text is drafted.
        application.add_handler(MessageHandler(
            filters.UpdateType.BUSINESS_MESSAGE,
            _on_business_message,
        ))
        application.add_handler(MessageHandler(
            filters.UpdateType.EDITED_BUSINESS_MESSAGE & (filters.TEXT | filters.CAPTION),
            _on_business_message,
        ))
        # Draft approval buttons — pattern-scoped so every other callback
        # falls through to the core adapter's CallbackQueryHandler.
        application.add_handler(CallbackQueryHandler(
            _on_draft_button, pattern=rf"^{CALLBACK_PREFIX}",
        ))
        # /biz command (owner-only state, adapter-level — no agent loop).
        application.add_handler(CommandHandler("biz", _on_biz_command))
        # Owner-DM edit capture in group -1 — runs before the core text
        # handler (default group 0); raises ApplicationHandlerStop only
        # when it consumes the message as an edit override.
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
            _on_owner_dm,
        ), group=-1)
        application.add_handler(MessageHandler(
            filters.VOICE & filters.ChatType.PRIVATE,
            _on_owner_voice,
        ), group=-1)

        logger.info("telegram-business: Business Mode handlers wired")

    ctx.register_telegram_handler(_wire)
