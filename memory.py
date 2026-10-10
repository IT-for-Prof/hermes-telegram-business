"""Optional contact memory for drafts, backed by a self-hosted Honcho workspace.

Enabled only when ``<HERMES_HOME>/telegram-business/honcho.json`` exists
(``{"baseUrl", "apiKey", "workspace"}``). Use a workspace of its own: contacts'
text must never reach the owner's main Hermes memory, where a tool-using agent
could treat it as instructions.

One Honcho session per chat; the owner is peer ``tgbiz_owner`` (not observed, so
business chats do not reshape the owner's model), each contact is peer
``tgbiz_<chat_id>`` (observed). Every call is synchronous and may raise; the
manager runs them off the event loop and treats failures as "no memory".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

OWNER_PEER = "tgbiz_owner"


def _safe_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-")[:200]


class ContactMemory:
    def __init__(self, client: Any, peer_config: Callable[..., Any]):
        self._honcho = client
        self._peer_config = peer_config      # honcho.api_types.SessionPeerConfig
        self._ready: set = set()

    @classmethod
    def from_file(cls, path: Path) -> Optional["ContactMemory"]:
        if not path.exists():
            return None
        cfg = json.loads(path.read_text())
        from honcho import Honcho
        from honcho.api_types import SessionPeerConfig

        return cls(Honcho(base_url=cfg["baseUrl"], api_key=cfg["apiKey"], workspace_id=cfg["workspace"],
                          timeout=10.0, max_retries=1), SessionPeerConfig)

    @staticmethod
    def contact_peer(conn_id: str, chat_id: str) -> str:
        return f"tgbiz_{_safe_id(conn_id)}_{_safe_id(chat_id)}"

    @staticmethod
    def _sid(conn_id: str, chat_id: str) -> str:
        return f"tgbiz-{_safe_id(conn_id)}-{_safe_id(chat_id)}"

    def _add(self, session: Any, conn_id: str, chat_id: str, turns: Iterable[Tuple[bool, str]]) -> None:
        owner, contact = self._honcho.peer(OWNER_PEER), self._honcho.peer(self.contact_peer(conn_id, chat_id))
        msgs = [(owner if is_owner else contact).message(text) for is_owner, text in turns if text]
        for i in range(0, len(msgs), 100):        # Honcho accepts at most 100 messages per batch
            session.add_messages(msgs[i:i + 100])

    def remember(self, conn_id: str, chat_id: str, is_owner: bool, text: str,
                 history: List[Tuple[bool, str]]) -> None:
        """Add one message. A new session is first given peers (before any message) and seeded with
        ``history``, the chat log as it was when this message was logged (it already contains the message)."""
        sid = self._sid(conn_id, chat_id)
        session = self._honcho.session(sid)
        if sid not in self._ready:
            session.add_peers([
                (OWNER_PEER, self._peer_config(observe_me=False, observe_others=False)),
                (self.contact_peer(conn_id, chat_id), self._peer_config(observe_me=True, observe_others=False)),
            ])
            self._ready.add(sid)
            if not session.messages(size=1).items:
                self._add(session, conn_id, chat_id, history)
                return
        self._add(session, conn_id, chat_id, [(is_owner, text)])

    def recall(self, conn_id: str, chat_id: str, query: str) -> str:
        """What Honcho derived about this contact: peer card, relevant conclusions, chat summary. No LLM call."""
        ctx = self._honcho.session(self._sid(conn_id, chat_id)).context(
            peer_target=self.contact_peer(conn_id, chat_id), search_query=query[:500] or None,
            tokens=800, summary=True)
        parts: List[str] = []
        if ctx.peer_card:
            parts += [f"- {c}" for c in ctx.peer_card]
        if ctx.peer_representation:
            parts.append(ctx.peer_representation.strip())
        if ctx.summary is not None and getattr(ctx.summary, "content", ""):
            parts.append("Earlier in this chat: " + ctx.summary.content.strip())
        return "\n".join(parts)[:3000]

    def forget(self, conn_id: str, chat_ids: Iterable[str]) -> None:
        """Delete each chat's session, every conclusion about the contact, and the contact's card."""
        for chat_id in chat_ids:
            sid = self._sid(conn_id, chat_id)
            try:
                self._honcho.session(sid).delete()
            except Exception as exc:              # already gone is fine; anything else is retried by the caller
                if type(exc).__name__ != "NotFoundError":
                    raise
            peer = self._honcho.peer(self.contact_peer(conn_id, chat_id))
            while True:
                page = peer.conclusions.list(size=50).items
                if not page:
                    break
                for c in page:
                    peer.conclusions.delete(c.id)
            peer.set_card([])
            self._ready.discard(sid)


def load(hermes_home: Path) -> Optional[ContactMemory]:
    return ContactMemory.from_file(hermes_home / "telegram-business" / "honcho.json")

