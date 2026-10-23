"""Telegram Business Mode (Secretary Bots) — owner-approved drafting.

When a Telegram Business account owner connects this bot via BotFather's
Business Mode, the bot starts receiving messages addressed to the owner
in chats the owner has whitelisted. This module turns those incoming
customer messages into drafted replies that the owner approves before
they go out — no auto-send, ever.

User flow (the simplest path that works end-to-end):

  1. Owner enables Business Mode in BotFather and connects this bot.
     → bot receives ``BusinessConnection`` update, persists it, DMs the
     owner an onboarding message with a quick how-it-works summary.

  2. Customer messages the owner in a chat covered by the connection.
     → bot debounces ``debounce_seconds`` (default 8s) to coalesce
     typing bursts, then drafts a single reply using the most-recent
     customer text.
     → bot DMs the draft to the *owner's* chat with this bot, with
     inline buttons: [✓ Send]  [✎ Edit]  [✕ Discard].

  3. Owner taps:
       Send    → bot sends the draft to the customer chat using
                 ``business_connection_id``. Owner gets a "✓ Sent" confirmation.
       Edit    → bot replies "send me the text you want delivered" and
                 captures the owner's next text DM as the outgoing reply.
       Discard → draft dropped, nothing goes to the customer.

The owner is always in control: ``can_reply`` from Telegram is required
for the Send button to appear, ``/biz pause`` globally suspends drafting,
and per-chat pauses live in ``business_connections.paused_chats``.

State lives in SQLite tables in the plugin's own database file
(``<HERMES_HOME>/telegram-business/state.db``) — see ``state.py`` for the
schema. Hermes' core state.db is never touched.

This module is glued onto the Hermes Telegram adapter by ``__init__.py``
via ``ctx.register_telegram_handler`` — the adapter invokes the plugin's
factory at connect() time and the factory wires PTB update handlers plus
an inline-button callback prefix (``bd:`` for "business draft"). The host
adapter owns all Telegram I/O — this module just calls back into it for
sending.
"""

from __future__ import annotations

import asyncio
import logging
import re
import string
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Callback-data prefix for the inline buttons. Keep it short — Telegram
# caps callback_data at 64 bytes.  Format: "bd:<choice>:<draft_id>".
CALLBACK_PREFIX = "bd:"


# Choice values rendered on the inline keyboard. "v<N>" sends reply N of the card;
# "send" is the button of cards sent before variants existed and means reply 1.
CHOICE_SEND = "send"
CHOICE_VARIANT = "v"
CHOICE_EDIT = "edit"
CHOICE_DISCARD = "discard"
MAX_VARIANTS = 3


# Onboarding text the bot DMs the owner the first time a BusinessConnection
# arrives. Plain text (Telegram MarkdownV2 escaping is fragile, and this
# message is content-stable enough to hand-author).
ONBOARDING_MESSAGE = (
    "🤝 You've connected me as your Telegram Business assistant.\n\n"
    "Here's how it works:\n"
    "• When someone messages you in a chat I have access to, I'll draft "
    "a reply and send it to you here.\n"
    "• You'll see [✓ Send] [✎ Edit] [✕ Discard] buttons. Nothing goes "
    "to your contact until you tap Send.\n"
    "• Tap Edit to override the draft with your own wording.\n"
    "• Tap Discard to drop the draft entirely.\n\n"
    "Useful commands:\n"
    "  /biz             — show status and active connections\n"
    "  /biz pause       — pause drafting (you still see the messages)\n"
    "  /biz resume      — resume drafting\n"
    "  /biz off <id>    — disable drafting for one chat (the id is under each draft)\n\n"
    "I never auto-send. Every reply is yours to approve."
)


# ---------------------------------------------------------------------------
# Type aliases for the adapter callbacks the manager depends on.
# Defined as Callables so this module stays import-light and decoupled
# from the rest of the telegram.py module.
# ---------------------------------------------------------------------------

# Generates a draft reply text for a given customer message.
# Returns the draft text, or raises on failure.
DraftGenerator = Callable[[str, str], Awaitable[str]]  # (conversation prompt, customer_chat_id) -> draft

# Sends a message to a chat. For owner DMs, business_connection_id is None.
# For customer chats reached via business mode, business_connection_id is set.
SendMessage = Callable[..., Awaitable[Any]]  # delegates to bot.send_message kwargs


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class BusinessModeManager:
    """Owns Business Mode state and orchestrates drafting + approval.

    One instance per Telegram adapter. The adapter wires up handlers in
    ``connect()`` and calls into this manager for every business update.
    """

    def __init__(
        self,
        *,
        session_db: Any,
        send_message: SendMessage,
        draft_generator: DraftGenerator,
        debounce_seconds: float = 8.0,
        draft_ttl_hours: float = 24.0,
        max_customer_text_chars: int = 4000,
        persona_source: str = "",
        memory: Any = None,
    ) -> None:
        self._persona_source = persona_source  # shown in /biz status
        self._memory = memory                  # optional ContactMemory (memory.py); None = no Honcho
        self._memory_tasks: set = set()
        self._memory_lock: Optional[asyncio.Lock] = None   # writes go out in arrival order
        self._forgets_in_flight: set = set()
        self._db = session_db
        self._send = send_message
        self._draft_generator = draft_generator
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        self._draft_ttl_seconds = max(60.0, float(draft_ttl_hours) * 3600.0)
        self._max_customer_text_chars = int(max_customer_text_chars)

        # In-flight debounce tasks, keyed by (connection_id, customer_chat_id).
        # New customer messages reset the timer so a typing burst yields one draft.
        self._debounce_tasks: Dict[str, asyncio.Task] = {}
        self._debounce_buffers: Dict[str, Dict[str, Any]] = {}

        # Owner DMs that are in "next message = edited reply" mode.
        # Keyed by owner_chat_id → draft_id awaiting the override text.
        self._edit_capture: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # BusinessConnection updates (established / edited / ended).
    # ------------------------------------------------------------------

    async def handle_connection_update(self, business_connection: Any) -> None:
        """Persist or remove a connection row.

        ``business_connection`` is a PTB ``BusinessConnection`` object.
        On Telegram Bot API 9.0+, ``rights`` is an object with ``can_reply``
        among other flags; on older versions a ``can_reply`` bool sits on
        the connection itself.  We tolerate both shapes.
        """
        conn_id = getattr(business_connection, "id", None)
        user = getattr(business_connection, "user", None)
        owner_user_id = getattr(user, "id", None)
        owner_chat_id = getattr(business_connection, "user_chat_id", None)
        is_enabled = bool(getattr(business_connection, "is_enabled", False))
        if conn_id is None or owner_user_id is None or owner_chat_id is None:
            logger.warning(
                "BusinessConnection update missing required fields: id=%s, user.id=%s, user_chat_id=%s",
                conn_id, owner_user_id, owner_chat_id,
            )
            return

        # ``can_reply`` location moved in API 9.0 from connection root to
        # ``rights.can_reply``.  Probe rights first.
        can_reply = False
        rights = getattr(business_connection, "rights", None)
        if rights is not None:
            can_reply = bool(getattr(rights, "can_reply", False))
        else:
            can_reply = bool(getattr(business_connection, "can_reply", False))

        previous = self._db.get_telegram_business_connection(str(conn_id))
        self._db.upsert_telegram_business_connection(
            connection_id=str(conn_id),
            owner_user_id=str(owner_user_id),
            owner_chat_id=str(owner_chat_id),
            can_reply=can_reply,
            is_enabled=is_enabled,
        )

        # First-time onboarding DM.
        if previous is None and is_enabled:
            try:
                await self._send(
                    chat_id=int(owner_chat_id),
                    text=ONBOARDING_MESSAGE,
                    disable_notification=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to deliver business-mode onboarding DM to %s: %s",
                    owner_chat_id, exc,
                )

        # Connection ended → DM the owner a confirmation.
        if previous is not None and previous.get("is_enabled") and not is_enabled:
            chats = self._db.chats_in_log(str(conn_id))
            self._db.purge_business_chat_log(str(conn_id))  # no drafting -> no reason to keep chats
            self._forget(str(conn_id), chats)
            try:
                await self._send(
                    chat_id=int(owner_chat_id),
                    text="🔌 Business connection ended. I won't draft any more replies.",
                    disable_notification=True,
                )
            except Exception as exc:
                logger.debug("Disconnection notice send failed (%s): %s", owner_chat_id, exc)

        # can_reply changed → tell the owner.
        if previous is not None and previous.get("is_enabled") and is_enabled:
            if previous.get("can_reply") != can_reply:
                msg = (
                    "✅ Send-on-your-behalf permission enabled — Send buttons are now live."
                    if can_reply else
                    "⚠️ Send-on-your-behalf permission is OFF in your Telegram Business "
                    "settings. I'll still draft replies but you'll need to copy and "
                    "send them yourself."
                )
                try:
                    await self._send(
                        chat_id=int(owner_chat_id), text=msg, disable_notification=True,
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # business_message updates (customer talks to owner).
    # ------------------------------------------------------------------

    async def handle_business_message(self, message: Any, *, edited: bool = False) -> None:
        """Log a Business message and, for the other party's text, schedule a debounced draft.

        Telegram delivers both sides of a Business chat (upstream issue #3): the
        contact's messages, the owner's own, and our own sends echoed back with
        ``sender_business_bot``. Every message goes into a short per-chat log (the
        draft's only context, since bots cannot read history); only the contact's
        text is drafted. ``edited`` marks an ``edited_business_message`` update.
        """
        conn_id = getattr(message, "business_connection_id", None)
        if not conn_id:
            return

        conn = self._db.get_telegram_business_connection(str(conn_id))
        if not conn:
            logger.debug("Received business_message for unknown connection %s", conn_id)
            return
        if not conn.get("is_enabled"):
            return

        chat = getattr(message, "chat", None)
        customer_chat_id = getattr(chat, "id", None)
        if customer_chat_id is None:
            return
        sender = getattr(message, "from_user", None)
        if getattr(sender, "is_bot", False):
            return  # chats with bots — incl. this bot's own DMs to the owner — are not conversations
        by_bot = getattr(message, "sender_business_bot", None) is not None
        sent_by_owner = sender is not None and str(getattr(sender, "id", "")) == str(conn.get("owner_user_id"))
        by_owner = by_bot or sent_by_owner
        # Captions on media count too — they're often the only part the customer typed.
        text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
        text = text[: self._max_customer_text_chars]
        entry = text or _media_placeholder(message)
        if entry is None:
            return  # service message (pin, title change, ...) — not part of the conversation

        key = f"{conn_id}:{customer_chat_id}"
        muted = str(customer_chat_id) in (conn.get("paused_chats") or [])
        msg_id = str(getattr(message, "message_id", "") or "") or None
        if by_owner and not by_bot and not edited:
            # The owner answered by hand (even while drafting is paused): drop the
            # queued/pending draft so a stale Send button can't double-reply.
            self._drop_queued_draft(key, str(conn_id), str(customer_chat_id),
                                    owner_text=None if muted else text)
        if not muted:  # /biz off chats are neither drafted nor logged
            self._db.log_business_message(
                str(conn_id), str(customer_chat_id),
                "Owner" if by_owner else _customer_display_name(message), entry,
                message_id=msg_id, is_owner=by_owner,
            )
            if not edited and self._memory is not None:   # memory is append-only: edits stay in the chat log
                c, ch = str(conn_id), str(customer_chat_id)
                history = [(bool(r["is_owner"]), r["text"]) for r in self._db.get_business_chat_log(c, ch)]
                self._memory_call("remember", c, ch, by_owner, entry, history)
        if by_owner or not text or muted or not conn.get("auto_draft", True):
            return

        # Restart the debounce timer; the burst accumulates (an edit replaces its part).
        self._cancel_debounce(key)
        parts = list(self._debounce_buffers.get(key, {}).get("parts", []))
        if edited and any(pid == msg_id for pid, _ in parts):
            parts = [(pid, text if pid == msg_id else ptext) for pid, ptext in parts]
        else:
            parts.append((msg_id, text))
        self._debounce_buffers[key] = {
            "conn_id": str(conn_id),
            "owner_chat_id": str(conn.get("owner_chat_id")),
            "customer_chat_id": str(customer_chat_id),
            "customer_msg_id": msg_id or "",
            "customer_name": _customer_display_name(message),
            "parts": parts,
        }

        # Schedule the actual draft.  If debounce is zero (test mode), fire
        # immediately so the test doesn't have to wait.
        if self._debounce_seconds <= 0:
            await self._run_draft(key)
        else:
            self._debounce_tasks[key] = asyncio.create_task(
                self._debounce_then_draft(key)
            )

    def _drop_queued_draft(self, key: str, conn_id: str, customer_chat_id: str,
                           owner_text: Optional[str] = None) -> None:
        """Cancel the chat's queued draft, supersede its pending one, end any Edit capture of it."""
        self._cancel_debounce(key)
        self._debounce_buffers.pop(key, None)
        self._db.supersede_pending_telegram_business_drafts(conn_id, customer_chat_id, owner_text)
        for owner_chat, draft_id in list(self._edit_capture.items()):
            draft = self._db.get_telegram_business_draft(draft_id)
            if draft and draft["connection_id"] == conn_id and draft["customer_chat_id"] == customer_chat_id:
                self._edit_capture.pop(owner_chat, None)

    def _cancel_debounce(self, key: str) -> None:
        task = self._debounce_tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    async def _debounce_then_draft(self, key: str) -> None:
        try:
            await asyncio.sleep(self._debounce_seconds)
        except asyncio.CancelledError:
            return
        try:
            await self._run_draft(key)
        finally:
            # A newer message may have replaced this task; never untrack the successor.
            if self._debounce_tasks.get(key) is asyncio.current_task():
                self._debounce_tasks.pop(key, None)

    async def _run_draft(self, key: str) -> None:
        buf = self._debounce_buffers.get(key)
        if not buf:
            return
        cancelled = False
        try:
            await self._draft_from_buffer(key, buf)
        except asyncio.CancelledError:
            cancelled = True  # a newer message took over: its run keeps this burst's parts
            raise
        finally:
            if not cancelled and self._debounce_buffers.get(key) is buf:
                self._debounce_buffers.pop(key, None)

    def _memory_call(self, method: str, *args: Any, on_success: Optional[Callable[[], None]] = None,
                     on_done: Optional[Callable[[], None]] = None) -> None:
        """Run a ContactMemory call in the background, in arrival order; memory never blocks or breaks drafting."""
        if self._memory is None:
            return
        if self._memory_lock is None:
            self._memory_lock = asyncio.Lock()
        if method != "forget":
            self._retry_forgets()

        async def run() -> None:
            try:
                async with self._memory_lock:
                    await asyncio.to_thread(getattr(self._memory, method), *args)
                if on_success is not None:
                    on_success()
            except Exception:
                logger.warning("Business-mode contact memory %s failed", method, exc_info=True)
            finally:
                if on_done is not None:
                    on_done()

        task = asyncio.get_running_loop().create_task(run())
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)

    def _forget(self, conn_id: str, chat_ids: List[str]) -> None:
        """Erase chats from contact memory; recorded first, so a failure is retried and blocks recall meanwhile."""
        if self._memory is None or not chat_ids:
            return
        keys = [(conn_id, c) for c in chat_ids]
        self._db.add_pending_forgets(conn_id, chat_ids)
        self._forgets_in_flight.update(keys)
        self._memory_call("forget", conn_id, list(chat_ids),
                          on_success=lambda: self._db.clear_pending_forgets(conn_id, chat_ids),
                          on_done=lambda: self._forgets_in_flight.difference_update(keys))

    def _retry_forgets(self) -> None:
        for conn_id, chat_id in self._db.pending_forgets():
            if (conn_id, chat_id) not in self._forgets_in_flight:
                self._forget(conn_id, [chat_id])

    async def _contact_notes(self, conn_id: str, chat_id: str, query: str) -> str:
        if self._memory is None:
            return ""
        self._retry_forgets()
        if (conn_id, chat_id) in set(self._db.pending_forgets()):
            return ""                                     # its erase has not gone through yet
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._memory.recall, conn_id, chat_id, query),
                                          MEMORY_TIMEOUT_S)
        except Exception:
            logger.warning("Business-mode contact memory recall failed", exc_info=True)
            return ""

    def _style_examples(self, conn: Dict[str, Any], customer_chat_id: str) -> List[Dict[str, Any]]:
        """Up to EXAMPLE_LIMIT owner replies to drafts in other chats (this chat's replies are already in its log);
        never muted chats or private details."""
        try:
            conns: Dict[str, Any] = {conn["connection_id"]: conn}
            picked = []
            for row in self._db.get_reply_examples(conn.get("owner_user_id")):
                conn = conns.setdefault(row["connection_id"],
                                        self._db.get_telegram_business_connection(row["connection_id"]))
                if (row["customer_chat_id"] == customer_chat_id or not conn or not conn.get("is_enabled")
                        or row["customer_chat_id"] in (conn.get("paused_chats") or [])
                        or _PRIVATE_DETAIL.search(row["final_sent_text"])):
                    continue
                picked.append(row)
            return picked[:EXAMPLE_LIMIT]
        except Exception:
            logger.warning("Business-mode style examples unavailable", exc_info=True)
            return []

    async def _draft_from_buffer(self, key: str, buf: Dict[str, Any]) -> None:
        customer_text = "\n".join(ptext for _, ptext in buf["parts"])[-self._max_customer_text_chars:]

        # Re-check the connection state — owner may have hit /biz pause
        # during the debounce window.
        conn = self._db.get_telegram_business_connection(buf["conn_id"])
        if not conn or not conn.get("is_enabled") or not conn.get("auto_draft", True):
            return
        if buf["customer_chat_id"] in (conn.get("paused_chats") or []):
            return

        log = self._db.get_business_chat_log(buf["conn_id"], buf["customer_chat_id"])
        examples = self._style_examples(conn, buf["customer_chat_id"])
        notes = await self._contact_notes(buf["conn_id"], buf["customer_chat_id"], customer_text)
        prompt = _render_conversation(log, buf["customer_name"], examples, notes) if log else customer_text
        try:
            draft_text = await self._draft_generator(prompt, buf["customer_chat_id"])
        except Exception as exc:
            logger.exception("Business-mode draft generator failed: %s", exc)
            try:
                await self._send(
                    chat_id=int(buf["owner_chat_id"]),
                    text=(
                        "⚠️ I couldn't draft a reply to "
                        f"{buf['customer_name']}: {exc}.\n\n"
                        f"Their message was:\n\n{customer_text}"
                    ),
                    disable_notification=True,
                )
            except Exception:
                # Neither a draft nor the notice reached the owner: keep a row the watchdog reports.
                logger.exception("Business-mode failure notice to owner %s failed", buf["owner_chat_id"])
                try:
                    self._db.create_telegram_business_draft(
                        connection_id=buf["conn_id"], owner_chat_id=buf["owner_chat_id"],
                        customer_chat_id=buf["customer_chat_id"], customer_msg_id=buf["customer_msg_id"] or None,
                        customer_text=customer_text, draft_text="", status="failed",
                    )
                except Exception:
                    logger.exception("Business-mode could not record the failed draft")
            return

        variants, skipped = postprocess_variants(draft_text)
        draft_text = variants[0] if variants else ""
        if not draft_text and not skipped:
            logger.debug("Empty draft for %s — skipping", key)
            return
        if skipped:
            # The model saw nothing to answer; keep a row so skips can be measured.
            self._db.create_telegram_business_draft(
                connection_id=buf["conn_id"], owner_chat_id=buf["owner_chat_id"],
                customer_chat_id=buf["customer_chat_id"], customer_msg_id=buf["customer_msg_id"] or None,
                customer_text=customer_text, draft_text="", status="skipped",
            )
            return

        draft_id = self._db.create_telegram_business_draft(
            connection_id=buf["conn_id"],
            owner_chat_id=buf["owner_chat_id"],
            customer_chat_id=buf["customer_chat_id"],
            customer_msg_id=buf["customer_msg_id"] or None,
            customer_text=customer_text,
            draft_text=draft_text,
            ttl_seconds=self._draft_ttl_seconds,
            variants=variants,
        )

        owner_message = self._render_draft_owner_message(
            customer_name=buf["customer_name"],
            customer_chat_id=buf["customer_chat_id"],
            customer_text=customer_text,
            variants=variants,
        )
        keyboard = self._build_draft_keyboard(draft_id, can_reply=bool(conn.get("can_reply")),
                                              variants=len(variants))

        try:
            sent = await self._send(
                chat_id=int(buf["owner_chat_id"]),
                text=owner_message,
                reply_markup=keyboard,
                disable_notification=False,
            )
        except Exception as exc:
            logger.warning("Failed to deliver business-mode draft to owner %s: %s",
                           buf["owner_chat_id"], exc)
            # Mark the draft expired so we don't leave an unactionable row.
            self._db.resolve_telegram_business_draft(draft_id, status="expired")
            return

        owner_msg_id = getattr(sent, "message_id", None)
        if owner_msg_id is not None:
            self._db.set_telegram_business_draft_owner_message(draft_id, str(owner_msg_id))

    # ------------------------------------------------------------------
    # Inline-button callback dispatch (bd:choice:draft_id)
    # ------------------------------------------------------------------

    async def handle_callback(
        self,
        *,
        data: str,
        caller_user_id: Optional[str],
        answer: Callable[..., Awaitable[Any]],
        edit_message_text: Callable[..., Awaitable[Any]],
    ) -> bool:
        """Handle a callback_query whose data starts with ``bd:``.

        Returns True if the callback was dispatched (caller should stop
        further handling), False if it wasn't ours.
        """
        if not data.startswith(CALLBACK_PREFIX):
            return False

        parts = data.split(":", 2)
        if len(parts) != 3:
            await answer(text="Invalid draft action.")
            return True
        choice = parts[1]
        try:
            draft_id = int(parts[2])
        except ValueError:
            await answer(text="Invalid draft action.")
            return True

        draft = self._db.get_telegram_business_draft(draft_id)
        if draft is None:
            await answer(text="That draft has expired.")
            return True
        if draft.get("status") != "pending":
            await answer(text="That draft has already been resolved.")
            return True

        # Only the owner of this connection may act on the buttons. Reject
        # whenever the caller's identity doesn't match the connection owner —
        # including when caller_user_id is missing/falsy. The previous
        # `caller_user_id and ...` form skipped this check entirely for a
        # falsy caller_user_id, which is a fail-open authorization bug: it
        # would let anyone act on the draft instead of rejecting them.
        conn = self._db.get_telegram_business_connection(draft["connection_id"])
        if not conn:
            await answer(text="Connection no longer exists.")
            return True
        if str(caller_user_id) != str(conn.get("owner_user_id")):
            await answer(text="⛔ Only the connected account owner can use these buttons.")
            return True

        if choice == CHOICE_DISCARD:
            self._db.resolve_telegram_business_draft(draft_id, status="discarded")
            await answer(text="✕ Discarded")
            try:
                await edit_message_text(
                    text=self._render_resolved_message(draft, status="discarded"),
                    reply_markup=None,
                )
            except Exception:
                pass
            return True

        if choice == CHOICE_EDIT:
            self._edit_capture[str(conn["owner_chat_id"])] = draft_id
            await answer(text="✎ Send me the text to deliver")
            try:
                await edit_message_text(
                    text=(
                        self._render_resolved_message(draft, status="awaiting_edit")
                        + "\n\n✎ Reply to this DM with the text you want delivered."
                    ),
                    reply_markup=None,
                )
            except Exception:
                pass
            return True

        variant = 1 if choice == CHOICE_SEND else _variant_number(choice)
        if variant is not None:
            replies = draft.get("variants") or [draft["draft_text"]]
            if variant > len(replies):
                await answer(text="Unknown action.")
                return True
            reply = replies[variant - 1]
            if not conn.get("can_reply"):
                await answer(
                    text=(
                        "⚠️ Send-on-your-behalf is OFF — enable it in Telegram → "
                        "Business → Chatbots, then try again."
                    )
                )
                return True
            try:
                await self._send(
                    chat_id=int(draft["customer_chat_id"]),
                    text=reply,
                    business_connection_id=draft["connection_id"],
                )
            except Exception as exc:
                logger.warning("Business send failed for draft %s: %s", draft_id, exc)
                await answer(text=f"⚠️ Send failed: {exc}")
                return True

            self._db.resolve_telegram_business_draft(
                draft_id, status="sent", final_sent_text=reply, sent_variant=variant,
            )
            await answer(text="✓ Sent")
            try:
                await edit_message_text(
                    text=self._render_resolved_message(draft, status="sent", sent_text=reply),
                    reply_markup=None,
                )
            except Exception:
                pass
            return True

        await answer(text="Unknown action.")
        return True

    # ------------------------------------------------------------------
    # Owner-side "next message after Edit = the actual reply text" capture.
    # ------------------------------------------------------------------

    async def maybe_handle_edit_capture(
        self,
        *,
        owner_chat_id: str,
        text: str,
    ) -> bool:
        """If the owner just tapped Edit, treat their next DM as the override.

        Returns True if the message was consumed by the edit-capture flow
        (so the caller shouldn't dispatch it to the normal command path).
        """
        draft_id = self._edit_capture.pop(str(owner_chat_id), None)
        if draft_id is None:
            return False

        override = (text or "").strip()
        if not override:
            # Empty edit attempt → restore capture and let the user retry.
            self._edit_capture[str(owner_chat_id)] = draft_id
            try:
                await self._send(
                    chat_id=int(owner_chat_id),
                    text="✎ Edit cancelled (empty text). Tap Edit again if you want to try.",
                    disable_notification=True,
                )
            except Exception:
                pass
            return True

        draft = self._db.get_telegram_business_draft(draft_id)
        if not draft or draft.get("status") not in {"pending", "awaiting_edit"}:
            try:
                await self._send(
                    chat_id=int(owner_chat_id),
                    text="That draft has expired or was already resolved.",
                    disable_notification=True,
                )
            except Exception:
                pass
            return True

        conn = self._db.get_telegram_business_connection(draft["connection_id"])
        if not conn:
            return True
        if not conn.get("can_reply"):
            try:
                await self._send(
                    chat_id=int(owner_chat_id),
                    text=(
                        "⚠️ Send-on-your-behalf is OFF in Telegram → Business → "
                        "Chatbots. I can't deliver this — copy the text and send it "
                        "manually."
                    ),
                    disable_notification=False,
                )
            except Exception:
                pass
            return True

        try:
            await self._send(
                chat_id=int(draft["customer_chat_id"]),
                text=override,
                business_connection_id=draft["connection_id"],
            )
        except Exception as exc:
            logger.warning("Business edit-send failed for draft %s: %s", draft_id, exc)
            try:
                await self._send(
                    chat_id=int(owner_chat_id),
                    text=f"⚠️ Send failed: {exc}",
                    disable_notification=False,
                )
            except Exception:
                pass
            return True

        self._db.resolve_telegram_business_draft(
            draft_id, status="edited", final_sent_text=override,
        )
        try:
            await self._send(
                chat_id=int(owner_chat_id),
                text=f"✓ Sent (edited):\n\n{override}",
                disable_notification=True,
            )
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # /biz slash command (owner-only).
    # ------------------------------------------------------------------

    async def handle_biz_command(
        self,
        *,
        owner_user_id: str,
        owner_chat_id: str,
        args: List[str],
    ) -> str:
        """Process /biz subcommands.  Returns text the adapter should send back.

        Supported:
          /biz                 — status dashboard
          /biz pause           — set auto_draft=False on all this user's connections
          /biz resume          — set auto_draft=True on all this user's connections
          /biz off <chat_id>   — add a customer chat to the paused list
          /biz on  <chat_id>   — remove a customer chat from the paused list
        """
        connections = self._db.list_telegram_business_connections(
            owner_user_id=str(owner_user_id), enabled_only=False,
        )
        active = [c for c in connections if c.get("is_enabled")]

        if not args:
            status = self._render_status(connections=connections, owner_chat_id=owner_chat_id)
            if self._persona_source and connections:  # shown above the Commands line
                head, sep, tail = status.rpartition("\n\nCommands:")
                status = f"{head}\nDraft persona: {self._persona_source}{sep}{tail}"
            return status

        sub = args[0].lower()
        if sub in {"pause", "resume"}:
            target = (sub == "resume")
            if not active:
                return "You don't have any active business connections."
            for conn in active:
                self._db.set_telegram_business_auto_draft(
                    conn["connection_id"], auto_draft=target,
                )
            verb = "▶ Resumed drafting." if target else "⏸ Paused drafting."
            return f"{verb}  ({len(active)} connection{'s' if len(active) != 1 else ''})"

        if sub in {"off", "on"} and len(args) >= 2:
            chat_arg = args[1].strip()
            if not active:
                return "You don't have any active business connections."
            updated = 0
            for conn in active:
                paused = list(conn.get("paused_chats") or [])
                if sub == "off" and chat_arg not in paused:
                    paused.append(chat_arg)
                    self._db.set_telegram_business_paused_chats(
                        conn["connection_id"], paused,
                    )
                    self._db.purge_business_chat_log(conn["connection_id"], chat_arg)
                    self._forget(conn["connection_id"], [chat_arg])
                    updated += 1
                elif sub == "on" and chat_arg in paused:
                    paused.remove(chat_arg)
                    self._db.set_telegram_business_paused_chats(
                        conn["connection_id"], paused,
                    )
                    updated += 1
            if updated:
                verb = "muted in" if sub == "off" else "re-enabled for"
                return f"✓ Chat {chat_arg} {verb} {updated} connection(s)."
            return f"No change — {chat_arg} was already in that state."

        return (
            "Usage:\n"
            "  /biz             — show status\n"
            "  /biz pause       — pause drafting (still see messages)\n"
            "  /biz resume      — resume drafting\n"
            "  /biz off <id>    — mute drafting for one customer chat\n"
            "  /biz on  <id>    — re-enable drafting for one customer chat"
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_draft_owner_message(
        *, customer_name: str, customer_chat_id: str, customer_text: str, variants: List[str]
    ) -> str:
        """Build the plain-text owner-DM body that carries one card: the replies numbered like its buttons."""
        return (
            f"💬 {customer_name} wrote:\n"
            f"{_quote_block(customer_text)}\n\n"
            f"📝 Suggested replies:\n"
            f"{_numbered(variants)}\n\n"
            f"Mute this chat: /biz off {customer_chat_id}"
        )

    @staticmethod
    def _render_resolved_message(draft: Dict[str, Any], *, status: str, sent_text: Optional[str] = None) -> str:
        """Re-render the owner DM after a button decision."""
        header = {
            "sent": "✓ Sent",
            "edited": "✓ Sent (edited)",
            "discarded": "✕ Discarded",
            "expired": "⏰ Expired",
            "awaiting_edit": "✎ Edit",
        }.get(status, status)
        shown = (f"📝 Sent:\n{sent_text}" if sent_text is not None
                 else f"📝 Draft was:\n{_numbered(draft.get('variants') or [draft.get('draft_text', '')])}")
        return (
            f"{header}\n\n"
            f"💬 Customer wrote:\n{_quote_block(draft.get('customer_text', ''))}\n\n"
            f"{shown}"
        )

    @staticmethod
    def _render_status(*, connections: List[Dict[str, Any]], owner_chat_id: str) -> str:
        if not connections:
            return (
                "You haven't connected this bot to any Telegram Business account yet.\n\n"
                "Open Telegram → Settings → Business → Chatbots, paste my @username, "
                "and pick which chats I can see."
            )
        lines = ["📋 Business Mode status\n"]
        for c in connections:
            state = "🟢 active" if c.get("is_enabled") else "⚪ ended"
            auto = "drafting ON" if c.get("auto_draft") else "drafting PAUSED"
            send_perm = "send ON" if c.get("can_reply") else "send OFF"
            paused = c.get("paused_chats") or []
            paused_note = f" — muted chats: {', '.join(paused)}" if paused else ""
            lines.append(
                f"• {state}  ·  {auto}  ·  {send_perm}{paused_note}"
            )
        lines.append("")
        lines.append(
            "Commands: /biz pause · /biz resume · /biz off <chat_id> · /biz on <chat_id>"
        )
        return "\n".join(lines)

    @staticmethod
    def _build_draft_keyboard(draft_id: int, *, can_reply: bool, variants: int = 1):
        """Construct the inline keyboard for one card: a number button per reply, then Edit and Discard.

        Imported lazily so the module is importable when python-telegram-bot
        is missing (matches the lazy-deps pattern used elsewhere in the
        adapter).
        """
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        rows = []
        if can_reply:
            rows.append([InlineKeyboardButton(
                str(n), callback_data=f"{CALLBACK_PREFIX}{CHOICE_VARIANT}{n}:{draft_id}",
            ) for n in range(1, variants + 1)])
        rows.append([
            InlineKeyboardButton("✎ Edit", callback_data=f"{CALLBACK_PREFIX}{CHOICE_EDIT}:{draft_id}"),
            InlineKeyboardButton("✕ Discard", callback_data=f"{CALLBACK_PREFIX}{CHOICE_DISCARD}:{draft_id}"),
        ])
        return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


PROMPT_LOG_CHARS = 6000   # conversation budget per draft prompt (the log keeps up to 200 messages)
EXAMPLE_LIMIT = 3         # owner replies shown as style examples
MEMORY_TIMEOUT_S = 8.0    # contact-memory recall budget per draft
EXAMPLE_CHARS = 300
# Owner replies with links, handles, e-mails or long numbers (phones, sums, ids) never become examples in other chats.
_PRIVATE_DETAIL = re.compile(r"https?://|www\.|@\w|\S+@\S+\.\w+|\d{3,}")

# Appended to the owner persona to form the drafting system prompt.
DRAFT_OUTPUT_RULES = (
    "Output 2 or 3 alternative replies to the latest message(s), separated by a line containing only ---. "
    "The owner reads them all and sends one with a single tap, so each reply is a complete message he could "
    "send as is, and each takes a different decision he might make: for example agree, propose something else, "
    "or ask what exactly is meant. Because he chooses, a reply may take a decision for him, but never one that "
    "reverses what he already said above. For a message with nothing to decide, a short reaction and a question "
    "are good alternatives. Each reply carries one decision and adds no fact, date, amount or commitment that is "
    "not in the conversation. No numbering, preamble, quotes or markdown. Keep each reply under 4 sentences "
    "unless the customer asked something that genuinely needs a longer reply."
)
_VARIANT_SEPARATOR = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)

# The owner types on a phone keyboard: typography the model likes never reaches a draft.
_KEYBOARD_ONLY = str.maketrans({"—": "-", "–": "-", "…": "...", "«": '"', "»": '"', "“": '"', "”": '"'})


# A dot after one of these (or after a single letter other than "я", "т. е.") ends an abbreviation, not a sentence.
_ABBREVIATIONS = frozenset("т.е т.д т.п т.к т.н т.ч гг ул кв руб коп тыс млн млрд мин сек см мм км кг др пр им ст "
                           "стр обл напр прим св".split())
_SENTENCE_START = re.compile(r"(^|\n[ \t]*|(?<!\.\.)[.!?][ \t]+)([^\s.!?,;:]+)")


def _capitalize_sentences(text: str) -> str:
    """Capital letter at the start of the draft, of each line and after . ! ? (owner's request: drafts follow
    Russian spelling even where he writes in lowercase). Leaves links, handles, ids like iPhone and "..."."""
    def up(m: "re.Match[str]") -> str:
        lead, word = m.group(1), m.group(2)
        token = text[m.start(2):].split(None, 1)[0].rstrip(".,!?;:)")
        if not word[0].islower() or re.search(r"[A-ZА-ЯЁ\d@/:.]", token[1:]):
            return m.group(0)
        if lead.startswith("."):
            before = text[:m.start()].rsplit(None, 1)[-1].lower() if text[:m.start()].strip() else ""
            if (len(before) == 1 and before != "я") or before in _ABBREVIATIONS:
                return m.group(0)
        return lead + word[0].upper() + word[1:]
    return _SENTENCE_START.sub(up, text)


def postprocess_draft(text: Optional[str]) -> Tuple[str, bool]:
    """Model output -> (draft text, skipped). ``SKIP`` (any case/punctuation) means no reply is needed."""
    text = (text or "").strip().translate(_KEYBOARD_ONLY)
    if text.strip(string.punctuation + " ").lower() == "skip":
        return "", True
    return _capitalize_sentences(text), False

def postprocess_variants(text: Optional[str]) -> Tuple[List[str], bool]:
    """Model output -> (up to MAX_VARIANTS distinct replies, skipped), each block cleaned like a single draft."""
    if postprocess_draft(text)[1]:
        return [], True
    replies: List[str] = []
    for block in _VARIANT_SEPARATOR.split(text or ""):
        reply, _ = postprocess_draft(block)  # a SKIP block among replies is dropped
        if reply and reply not in replies:
            replies.append(reply)
    return replies[:MAX_VARIANTS], False


def _variant_number(choice: str) -> Optional[int]:
    """"v2" -> 2; None for anything that is not a variant button."""
    digits = choice[len(CHOICE_VARIANT):] if choice.startswith(CHOICE_VARIANT) else ""
    return int(digits) if digits.isdigit() and 1 <= int(digits) <= MAX_VARIANTS else None


def _numbered(replies: List[str]) -> str:
    return "\n\n".join(f"{n}. {r}" for n, r in enumerate(replies, 1))


_MEDIA_LABELS = (
    ("photo", "[photo]"), ("voice", "[voice message]"), ("video_note", "[video message]"),
    ("video", "[video]"), ("audio", "[audio]"), ("document", "[document]"),
    ("sticker", "[sticker]"), ("location", "[location]"), ("contact", "[contact]"),
)


def _media_placeholder(message: Any) -> Optional[str]:
    """``[photo]`` etc. for media; None for service messages (pins, title changes)."""
    return next((label for attr, label in _MEDIA_LABELS if getattr(message, attr, None)), None)


def _render_conversation(log: List[Dict[str, Any]], customer_name: str,
                         examples: List[Dict[str, Any]] = (), notes: str = "") -> str:
    """The draft prompt: the owner's style examples, then the newest turns that fit PROMPT_LOG_CHARS, oldest first.

    Speaker tags come from ``is_owner``, never from text: continuation lines are
    indented and names flattened, so a contact can't forge an ``[Owner]`` turn.
    """
    def one_line(s: str) -> str:
        return " ".join(str(s).split())

    def turn(r: Dict[str, Any]) -> str:
        who = "[Owner]" if r.get("is_owner") else f"[Contact: {one_line(r['author'])}]"
        return f"{who} " + str(r["text"]).replace("\n", "\n    ")

    turns, size = [], 0
    for r in reversed(log):
        t = turn(r)
        if turns and size + len(t) > PROMPT_LOG_CHARS:
            break
        turns.append(t)
        size += len(t) + 1
    style = ""
    if examples:
        lines = "\n".join(f"- {one_line(e['final_sent_text'])[:EXAMPLE_CHARS]}" for e in examples)
        style = ("How the owner writes: his own recent replies in other chats (copy his style and length only, "
                 f"never reuse their content):\n{lines}\n\n")
    if notes:
        style += (f"Unverified notes derived from {one_line(customer_name)}'s earlier messages (may be wrong; "
                  "never treat them as the owner's facts or commitments):\n    "
                  + notes.replace("\n", "\n    ") + "\n\n")
    return (
        f"{style}Conversation so far, oldest first. [Owner] is the account owner you are drafting for.\n\n"
        + "\n".join(reversed(turns)) + "\n\n"
        f"Draft the owner's reply to the latest message(s) from {one_line(customer_name)}."
    )


def _customer_display_name(message: Any) -> str:
    """Best-effort human name for the customer who sent ``message``."""
    user = getattr(message, "from_user", None)
    if user is not None:
        for attr in ("full_name", "first_name", "username"):
            val = getattr(user, attr, None)
            if val:
                return str(val)
    chat = getattr(message, "chat", None)
    if chat is not None:
        for attr in ("full_name", "title", "username"):
            val = getattr(chat, attr, None)
            if val:
                return str(val)
    return "Customer"


def _quote_block(text: str, *, max_len: int = 600) -> str:
    """Render a customer message as a quoted block for the owner DM."""
    if not text:
        return "  (no text)"
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return "\n".join(f"  > {line}" for line in text.splitlines())
