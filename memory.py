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

        return cls(Honcho(base_url=cfg["baseUrl"], api_key=cfg["apiKey"], workspace_id=cfg["workspace"]),
                   SessionPeerConfig)

    @staticmethod
    def contact_peer(chat_id: str) -> str:
        return f"tgbiz_{_safe_id(chat_id)}"

    def _session(self, conn_id: str, chat_id: str, history: Callable[[], Iterable[Tuple[bool, str]]]):
        sid = f"tgbiz-{_safe_id(conn_id)}-{_safe_id(chat_id)}"
        session = self._honcho.session(sid)
        if sid not in self._ready:
            # Peers and their observation settings must exist before the first message is added.
            session.add_peers([
                (OWNER_PEER, self._peer_config(observe_me=False, observe_others=False)),
                (self.contact_peer(chat_id), self._peer_config(observe_me=True, observe_others=False)),
            ])
            if not session.messages(size=1).items:      # new session: seed it with what the chat log already holds
                self._add(session, chat_id, history())
            self._ready.add(sid)
            return session, True
        return session, False

    def _add(self, session: Any, chat_id: str, turns: Iterable[Tuple[bool, str]]) -> None:
        msgs = [self._honcho.peer(OWNER_PEER if is_owner else self.contact_peer(chat_id)).message(text)
                for is_owner, text in turns if text]
        if msgs:
            session.add_messages(msgs)

    def remember(self, conn_id: str, chat_id: str, is_owner: bool, text: str,
                 history: Callable[[], Iterable[Tuple[bool, str]]]) -> None:
        session, seeded = self._session(conn_id, chat_id, history)
        if not seeded:                            # a seeded session already contains this message
            self._add(session, chat_id, [(is_owner, text)])

    def recall(self, conn_id: str, chat_id: str, query: str) -> str:
        """What Honcho derived about this contact: peer card, relevant conclusions, chat summary. No LLM call."""
        session, _ = self._session(conn_id, chat_id, lambda: [])
        ctx = session.context(peer_target=self.contact_peer(chat_id), search_query=query[:500] or None,
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
        for chat_id in chat_ids:
            sid = f"tgbiz-{_safe_id(conn_id)}-{_safe_id(chat_id)}"
            self._honcho.session(sid).delete()
            self._honcho.peer(self.contact_peer(chat_id)).set_card([])
            self._ready.discard(sid)


def load(hermes_home: Path) -> Optional[ContactMemory]:
    return ContactMemory.from_file(hermes_home / "telegram-business" / "honcho.json")

