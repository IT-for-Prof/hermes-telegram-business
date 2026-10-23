"""Plugin-owned SQLite state for Telegram Business Mode.

Two tables, both living in the plugin's own database file
(``<HERMES_HOME>/telegram-business/state.db``) — the plugin never touches
Hermes' core ``state.db`` schema:

  business_connections — one row per Telegram Business account that linked
    the bot via BotFather Business Mode. Updated by BusinessConnection
    updates (established / edited / ended).

  business_drafts — pending owner-approval drafts. Created when a customer
    messages a connected chat and the manager produces a candidate reply;
    resolved when the owner taps Send / Edit / Discard or when it expires.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS business_connections (
    connection_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    owner_chat_id TEXT NOT NULL,
    can_reply INTEGER NOT NULL DEFAULT 0,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    auto_draft INTEGER NOT NULL DEFAULT 1,
    paused_chats TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_biz_conn_owner
    ON business_connections(owner_user_id);

CREATE TABLE IF NOT EXISTS business_drafts (
    draft_id INTEGER PRIMARY KEY AUTOINCREMENT,
    connection_id TEXT NOT NULL,
    owner_chat_id TEXT NOT NULL,
    customer_chat_id TEXT NOT NULL,
    customer_msg_id TEXT,
    customer_text TEXT NOT NULL,
    draft_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    owner_message_id TEXT,
    final_sent_text TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_biz_drafts_conn_customer
    ON business_drafts(connection_id, customer_chat_id, status);

-- Short rolling log per chat (both sides): the only conversation context a
-- Business bot can have, since bots cannot read chat history.
CREATE TABLE IF NOT EXISTS business_chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    connection_id TEXT NOT NULL,
    customer_chat_id TEXT NOT NULL,
    author TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    message_id TEXT,
    is_owner INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS business_pending_forgets (
    connection_id TEXT NOT NULL,
    customer_chat_id TEXT NOT NULL,
    PRIMARY KEY (connection_id, customer_chat_id)
);

CREATE INDEX IF NOT EXISTS idx_biz_log_chat
    ON business_chat_log(connection_id, customer_chat_id, id);
"""

# Added after the first release; ALTER TABLE adds them to older databases on open.
#   variants      JSON list of the card's replies (draft_text keeps the first)
#   origin        NULL for a card; gist_card / gist_dm / gist_digest for a gist reply
#   sent_variant  the number of the variant the owner sent
#   gist          the owner's gist as typed, transcribed or passed to the tool
#   card_id       on a gist row: the chat's card it closes as ``gisted`` once sent
_ADDED_DRAFT_COLUMNS = (("variants", "TEXT"), ("origin", "TEXT"), ("sent_variant", "INTEGER"),
                        ("gist", "TEXT"), ("card_id", "INTEGER"))


class BusinessStateDB:
    """Thread-safe SQLite store for connections + drafts."""

    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            have = {r[1] for r in self._conn.execute("PRAGMA table_info(business_drafts)")}
            for name, kind in _ADDED_DRAFT_COLUMNS:
                if name not in have:
                    self._conn.execute(f"ALTER TABLE business_drafts ADD COLUMN {name} {kind}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def upsert_telegram_business_connection(
        self,
        *,
        connection_id: str,
        owner_user_id: str,
        owner_chat_id: str,
        can_reply: bool,
        is_enabled: bool,
    ) -> None:
        """Insert or update a connection row.

        Preserves ``auto_draft`` and ``paused_chats`` across updates so the
        owner's preferences survive Telegram re-issuing the connection.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO business_connections (
                    connection_id, owner_user_id, owner_chat_id,
                    can_reply, is_enabled, auto_draft, paused_chats,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, '[]', ?, ?)
                ON CONFLICT(connection_id) DO UPDATE SET
                    owner_user_id = excluded.owner_user_id,
                    owner_chat_id = excluded.owner_chat_id,
                    can_reply = excluded.can_reply,
                    is_enabled = excluded.is_enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    str(connection_id), str(owner_user_id), str(owner_chat_id),
                    1 if can_reply else 0, 1 if is_enabled else 0, now, now,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _conn_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["paused_chats"] = json.loads(d.get("paused_chats") or "[]")
        except (TypeError, ValueError):
            d["paused_chats"] = []
        d["can_reply"] = bool(d.get("can_reply"))
        d["is_enabled"] = bool(d.get("is_enabled"))
        d["auto_draft"] = bool(d.get("auto_draft", 1))
        return d

    def get_telegram_business_connection(
        self, connection_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM business_connections WHERE connection_id = ?",
                (str(connection_id),),
            ).fetchone()
        return self._conn_row_to_dict(row) if row is not None else None

    def list_telegram_business_connections(
        self, *, owner_user_id: Optional[str] = None, enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM business_connections"
        clauses: List[str] = []
        params: List[Any] = []
        if owner_user_id is not None:
            clauses.append("owner_user_id = ?")
            params.append(str(owner_user_id))
        if enabled_only:
            clauses.append("is_enabled = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._conn_row_to_dict(r) for r in rows]

    def set_telegram_business_auto_draft(
        self, connection_id: str, *, auto_draft: bool
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE business_connections SET auto_draft = ?, updated_at = ? "
                "WHERE connection_id = ?",
                (1 if auto_draft else 0, time.time(), str(connection_id)),
            )
            self._conn.commit()

    def set_telegram_business_paused_chats(
        self, connection_id: str, paused_chats: List[str]
    ) -> None:
        payload = json.dumps([str(c) for c in (paused_chats or [])])
        with self._lock:
            self._conn.execute(
                "UPDATE business_connections SET paused_chats = ?, updated_at = ? "
                "WHERE connection_id = ?",
                (payload, time.time(), str(connection_id)),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Drafts
    # ------------------------------------------------------------------

    def create_telegram_business_draft(
        self,
        *,
        connection_id: str,
        owner_chat_id: str,
        customer_chat_id: str,
        customer_msg_id: Optional[str],
        customer_text: str,
        draft_text: str,
        ttl_seconds: float = 86400.0,
        status: str = "pending",
        variants: Optional[List[str]] = None,
        origin: Optional[str] = None,
        gist: Optional[str] = None,
        card_id: Optional[int] = None,
    ) -> int:
        """Insert a pending draft row and return its draft_id.

        Any prior pending drafts for the same (connection, customer_chat)
        are marked superseded so only one Send button is ever live per
        conversation. A gist row (``origin`` set) supersedes only earlier
        gist rows: the card stays pending, buttonless, until the gist is sent.
        ``status="skipped"`` (no reply needed) and ``"copy"`` leave them alone.
        """
        now = time.time()
        with self._lock:
            if status == "pending":
                self._supersede_pending(now, connection_id, customer_chat_id, gists_only=origin is not None)
            cur = self._conn.execute(
                """
                INSERT INTO business_drafts (
                    connection_id, owner_chat_id, customer_chat_id,
                    customer_msg_id, customer_text, draft_text,
                    status, created_at, updated_at, expires_at,
                    variants, origin, gist, card_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(connection_id), str(owner_chat_id), str(customer_chat_id),
                    str(customer_msg_id) if customer_msg_id is not None else None,
                    customer_text, draft_text, status,
                    now, now, now + max(60.0, float(ttl_seconds)),
                    json.dumps(variants, ensure_ascii=False) if variants else None, origin, gist, card_id,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def log_business_message(
        self, connection_id: str, customer_chat_id: str, author: str, text: str, *,
        message_id: Optional[str] = None, is_owner: bool = False, keep: int = 200,
    ) -> None:
        """Record one message of a chat (an edit replaces its row) and keep the newest ``keep``."""
        conn_id, chat_id = str(connection_id), str(customer_chat_id)
        with self._lock:
            updated = message_id is not None and self._conn.execute(
                "UPDATE business_chat_log SET text = ? WHERE connection_id = ? "
                "AND customer_chat_id = ? AND message_id = ?",
                (text, conn_id, chat_id, str(message_id)),
            ).rowcount
            if not updated:
                self._conn.execute(
                    "INSERT INTO business_chat_log (connection_id, customer_chat_id, author, text, "
                    "created_at, message_id, is_owner) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (conn_id, chat_id, author, text, time.time(),
                     None if message_id is None else str(message_id), int(is_owner)),
                )
                self._conn.execute(
                    "DELETE FROM business_chat_log WHERE connection_id = ? AND customer_chat_id = ? "
                    "AND id NOT IN (SELECT id FROM business_chat_log WHERE connection_id = ? "
                    "AND customer_chat_id = ? ORDER BY id DESC LIMIT ?)",
                    (conn_id, chat_id, conn_id, chat_id, int(keep)),
                )
            self._conn.commit()

    def get_business_chat_log(self, connection_id: str, customer_chat_id: str) -> List[Dict[str, Any]]:
        """The chat's logged messages, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT author, text, is_owner FROM business_chat_log WHERE connection_id = ? "
                "AND customer_chat_id = ? ORDER BY id",
                (str(connection_id), str(customer_chat_id)),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_reply_examples(self, owner_user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """This owner's latest replies that replaced a draft (``replied``/``edited``), newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.connection_id, d.customer_chat_id, d.final_sent_text FROM business_drafts d "
                "JOIN business_connections c ON c.connection_id = d.connection_id "
                "WHERE c.owner_user_id = ? AND d.status IN ('replied', 'edited') "
                "AND coalesce(d.final_sent_text, '') != '' AND d.draft_text != '' AND d.final_sent_text != d.draft_text "
                "ORDER BY d.updated_at DESC LIMIT ?",
                (str(owner_user_id), int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_pending_forgets(self, connection_id: str, chat_ids: List[str]) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR IGNORE INTO business_pending_forgets VALUES (?, ?)",
                                   [(str(connection_id), str(c)) for c in chat_ids])
            self._conn.commit()

    def clear_pending_forgets(self, connection_id: str, chat_ids: List[str]) -> None:
        with self._lock:
            self._conn.executemany("DELETE FROM business_pending_forgets WHERE connection_id = ? AND customer_chat_id = ?",
                                   [(str(connection_id), str(c)) for c in chat_ids])
            self._conn.commit()

    def pending_forgets(self) -> List[Tuple[str, str]]:
        with self._lock:
            return [tuple(r) for r in self._conn.execute(
                "SELECT connection_id, customer_chat_id FROM business_pending_forgets ORDER BY 1, 2").fetchall()]

    def last_contact_message_at(self, connection_id: str, customer_chat_id: str) -> Optional[float]:
        """When the contact last wrote in this chat (Telegram's 24-hour reply window), None if never."""
        with self._lock:
            row = self._conn.execute(
                "SELECT max(created_at) FROM business_chat_log WHERE connection_id = ? "
                "AND customer_chat_id = ? AND is_owner = 0",
                (str(connection_id), str(customer_chat_id)),
            ).fetchone()
        return row[0]

    def find_chats(self, owner_user_id: str, query: str) -> List[Dict[str, Any]]:
        """This owner's logged chats whose id equals ``query`` or whose contact name contains it (any case)."""
        q = str(query).strip()
        with self._lock:
            rows = self._conn.execute(
                "SELECT l.connection_id, l.customer_chat_id, l.author FROM business_chat_log l "
                "JOIN business_connections c ON c.connection_id = l.connection_id "
                "WHERE c.owner_user_id = ? AND l.id = (SELECT max(id) FROM business_chat_log "
                "  WHERE connection_id = l.connection_id AND customer_chat_id = l.customer_chat_id AND is_owner = 0)",
                (str(owner_user_id),),
            ).fetchall()
        return [{"connection_id": r[0], "customer_chat_id": r[1], "name": r[2]} for r in rows
                if q and (r[1] == q or q.casefold() in r[2].casefold())]

    def chats_in_log(self, connection_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT customer_chat_id FROM business_chat_log WHERE connection_id = ?",
                                      (str(connection_id),)).fetchall()
        return sorted(r[0] for r in rows)

    def purge_business_chat_log(self, connection_id: str, customer_chat_id: Optional[str] = None) -> None:
        """Forget a chat's log and its draft texts (or those of every chat of a connection).

        A pending draft is expired too: the chat is muted or the connection is gone.
        """
        where, args = "connection_id = ?", (str(connection_id),)
        if customer_chat_id is not None:
            where, args = "connection_id = ? AND customer_chat_id = ?", args + (str(customer_chat_id),)
        with self._lock:
            self._conn.execute(f"DELETE FROM business_chat_log WHERE {where}", args)
            self._conn.execute(
                "UPDATE business_drafts SET customer_text = '', draft_text = '', final_sent_text = NULL, "
                "variants = NULL, gist = NULL, "
                f"status = CASE WHEN status = 'pending' THEN 'expired' ELSE status END WHERE {where}", args)
            self._conn.commit()

    def supersede_pending_telegram_business_drafts(
        self, connection_id: str, customer_chat_id: str, owner_text: Optional[str] = None,
    ) -> None:
        """Retire the live draft of one conversation: the owner answered by hand (status ``replied``).

        ``owner_text`` (his reply) is kept as the draft's ``final_sent_text``, a
        "draft -> what he actually wrote" example, but only on drafts that have
        not expired. Without a pending draft it extends the latest reply recorded
        in the last 120 s, unless the contact has written since, so a reply typed
        as several messages stays whole.
        """
        now, conn_id, chat_id = time.time(), str(connection_id), str(customer_chat_id)
        with self._lock:
            if owner_text:
                captured = self._conn.execute(
                    "UPDATE business_drafts SET final_sent_text = ? WHERE connection_id = ? "
                    "AND customer_chat_id = ? AND status = 'pending' AND expires_at >= ?",
                    (owner_text, conn_id, chat_id, now),
                ).rowcount
                if not captured:
                    self._conn.execute(
                        "UPDATE business_drafts SET final_sent_text = final_sent_text || char(10) || ? "
                        "WHERE draft_id = (SELECT max(d.draft_id) FROM business_drafts d "
                        "  WHERE d.connection_id = ? AND d.customer_chat_id = ? AND d.status = 'replied' "
                        "  AND d.final_sent_text IS NOT NULL AND d.updated_at >= ? AND NOT EXISTS ("
                        "    SELECT 1 FROM business_chat_log l WHERE l.connection_id = d.connection_id "
                        "    AND l.customer_chat_id = d.customer_chat_id AND l.is_owner = 0 "
                        "    AND l.created_at > d.updated_at))",
                        (owner_text, conn_id, chat_id, now - 120),
                    )
            self._supersede_pending(now, conn_id, chat_id, status="replied")
            self._conn.commit()

    def _supersede_pending(self, now: float, connection_id: str, customer_chat_id: str,
                           status: str = "superseded", gists_only: bool = False) -> None:
        """Caller holds ``self._lock`` and commits."""
        self._conn.execute(
            "UPDATE business_drafts SET status = ?, updated_at = ? "
            "WHERE connection_id = ? AND customer_chat_id = ? AND status = 'pending'"
            + (" AND origin IS NOT NULL" if gists_only else ""),
            (status, now, str(connection_id), str(customer_chat_id)),
        )

    def get_telegram_business_draft(self, draft_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM business_drafts WHERE draft_id = ?",
                (int(draft_id),),
            ).fetchone()
        if row is None:
            return None
        draft = dict(row)
        draft["variants"] = json.loads(draft["variants"]) if draft["variants"] else None
        return draft

    def set_telegram_business_draft_owner_message(
        self, draft_id: int, owner_message_id: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE business_drafts SET owner_message_id = ?, updated_at = ? "
                "WHERE draft_id = ?",
                (str(owner_message_id), time.time(), int(draft_id)),
            )
            self._conn.commit()

    def resolve_telegram_business_draft(
        self,
        draft_id: int,
        *,
        status: str,
        final_sent_text: Optional[str] = None,
        sent_variant: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Atomically mark a draft sent / edited / discarded / expired.

        Returns the prior row, or None if the draft no longer exists or was
        already resolved (so callbacks for stale buttons no-op safely).
        A gist row that goes out closes its card as ``gisted``.
        """
        if status not in {"sent", "edited", "discarded", "expired"}:
            raise ValueError(f"invalid business draft status: {status}")
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM business_drafts WHERE draft_id = ? AND status = 'pending'",
                (int(draft_id),),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE business_drafts SET status = ?, "
                "final_sent_text = COALESCE(?, final_sent_text), "
                "sent_variant = COALESCE(?, sent_variant), "
                "updated_at = ? WHERE draft_id = ?",
                (status, final_sent_text, sent_variant, now, int(draft_id)),
            )
            if status in {"sent", "edited"} and row["card_id"] is not None:
                self._conn.execute(
                    "UPDATE business_drafts SET status = 'gisted', updated_at = ? "
                    "WHERE draft_id = ? AND status = 'pending'",
                    (now, int(row["card_id"])),
                )
            self._conn.commit()
            return dict(row)

    def get_pending_telegram_business_drafts_for_owner(
        self, owner_chat_id: str
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM business_drafts WHERE owner_chat_id = ? "
                "AND status = 'pending' ORDER BY created_at ASC",
                (str(owner_chat_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def expire_telegram_business_drafts(self, *, now: Optional[float] = None) -> int:
        cutoff = now if now is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE business_drafts SET status = 'expired', updated_at = ? "
                "WHERE status = 'pending' AND expires_at < ?",
                (cutoff, cutoff),
            )
            self._conn.commit()
            return int(cur.rowcount)
