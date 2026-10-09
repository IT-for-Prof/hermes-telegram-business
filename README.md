# hermes-telegram-business

**Observe-with-approval Telegram Business Mode (secretary bot) for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Connect your Telegram Business account to your Hermes bot and it becomes your secretary: every incoming customer message gets an LLM-drafted reply delivered to **your** DM with inline buttons — **nothing reaches the customer until you tap Send.**

```
Customer → your Business account (via Business Mode)
            ↓ debounce 8s, draft once
Bot → your DM
       💬 Carol wrote:
         > Hey, are you free Friday afternoon?

       📝 Suggested reply:
       Friday afternoon works — anywhere between 2 and 5pm?

       [✓ Send]   [✎ Edit]   [✕ Discard]
            ↓
Send    → delivered to the customer via your Business connection
Edit    → your next DM text becomes the outgoing message
Discard → dropped, the customer sees nothing
```

## Safety properties

- **No auto-send exists.** Every reply requires an owner button tap, even when `can_reply` is granted.
- **Owner-only buttons.** Callbacks are authorized against the connection's `owner_user_id`; anyone else gets `⛔`.
- **Drafts expire after 24h** (configurable). Stale buttons no-op.
- **Without `can_reply`, the Send button is hidden** — you're told to copy/paste manually instead of getting a button that silently fails.
- **Typing bursts coalesce** — messages within the debounce window accumulate into one draft (an edit replaces its part), one draft per coherent thought.
- **Only the other party is drafted.** Telegram delivers both sides of a Business chat: the owner's own messages and the bot's Send/Edit echoes are logged as context but never drafted. When the owner answers by hand, the chat's queued/pending draft is withdrawn so a stale Send can't double-reply. Bot senders are ignored.

## Requirements

- Hermes Agent with plugin Telegram-handler support (`ctx.register_telegram_handler`, hermes-agent PR #59159 / v0.18+)
- A Telegram **Business** subscription on your personal account
- Your Hermes Telegram bot with **Business Mode** toggled on in [@BotFather](https://t.me/BotFather)
- python-telegram-bot ≥ 21.1 (already installed with the Hermes Telegram platform)

## Install

```bash
git clone https://github.com/NousResearch/hermes-telegram-business \
    ~/.hermes/plugins/telegram-business
hermes plugins enable telegram-business
```

Restart the gateway (`hermes gateway restart`). Then:

1. @BotFather → your bot → **Business Mode** → enable.
2. Telegram → **Settings → Business → Chatbots** → add your bot → pick which chats it can see, and whether it can **reply on your behalf** (needed for the Send button).
3. Message your bot `/biz` to confirm the connection is live.

## Owner controls

| Command | Effect |
|---|---|
| `/biz` | status dashboard |
| `/biz pause` / `/biz resume` | global drafting kill switch |
| `/biz off <chat_id>` / `/biz on <chat_id>` | per-customer-chat mute |

## Configuration

Optional block in `~/.hermes/config.yaml`:

```yaml
plugins:
  entries:
    telegram-business:
      settings:
        debounce_seconds: 8        # coalesce typing bursts
        draft_ttl_hours: 24        # stale-draft expiry
        max_customer_text_chars: 4000
        owner_persona: >
          You are drafting replies for a freelance photographer.
          Friendly, brief, always suggest a concrete next step.
```

Drafting uses your active Hermes model through the host-owned plugin LLM surface (`ctx.llm`) — no separate API key.

## State

Plugin-owned SQLite at `~/.hermes/telegram-business/state.db` (three tables: connections, drafts, and `business_chat_log` — the 20 newest messages of each chat, both sides, used as draft context; purged on `/biz off <chat_id>` and when the connection ends). Hermes' core state is never touched. Delete the file to reset.

## v1 limits

- **No vision** — media enters the context as `[photo]`, `[voice message]`, ... placeholders; captions are drafted like text.
- **History starts at connection time** — bots can't read older messages; the draft sees the last 20 messages logged since.
- **No persona learning from edits** — your overrides go to the customer but don't train future drafts.

## Tests

```bash
python3 -m pytest
```

63 tests covering connection lifecycle, owner/bot message handling, the chat log, edits, config, draft supersession, debounce coalescing, all three button paths, edit capture, owner-scoped callback authorization, and `/biz` subcommands. No network, no live Telegram.

## Credits

Based on the design and implementation from [hermes-agent#30055](https://github.com/NousResearch/hermes-agent/pull/30055), replatformed as a standalone plugin. Related earlier community proposals: [#26654](https://github.com/NousResearch/hermes-agent/pull/26654) by @evgyur (earliest submission), [#35342](https://github.com/NousResearch/hermes-agent/pull/35342) by @MilekhinAV, [#46728](https://github.com/NousResearch/hermes-agent/pull/46728) by @kxnkxv.

## License

MIT
