# hermes-telegram-business

**Observe-with-approval Telegram Business Mode (secretary bot) for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Connect your Telegram Business account to your Hermes bot and it becomes your secretary: every incoming customer message gets 2–3 LLM-drafted replies, each a different decision, delivered to **your** DM with inline buttons — **nothing reaches the customer until you tap one.**

```
Customer → your Business account (via Business Mode)
            ↓ debounce (default 8s), draft once
Bot → your DM
       💬 Carol wrote:
         > Hey, are you free Friday afternoon?

       📝 Suggested replies:
       1. Friday works, what time suits you?

       2. Friday is busy, how about Monday?

       3. Which Friday do you mean, this week?

       [1] [2] [3]
       [✍ Своя суть]  [✎ Edit]  [✕ Discard]
            ↓
1 / 2 / 3   → that reply is delivered via your Business connection
Своя суть   → say or type the gist ("monday 3pm"); the bot words it as your reply
              and shows it with Send / Edit / Discard
Edit        → your next DM text becomes the outgoing message
Discard     → dropped, the customer sees nothing
```

The gist also works without a card: tell the Hermes agent in the bot DM "Carol: monday 3pm" (typed or as a voice note) and it calls the plugin's `secretary_reply` tool, which shows the worded reply the same way, or asks which chat you mean when the name fits several. Outside Telegram's 24-hour reply window the reply is shown for copying instead of with Send.

## Safety properties

- **No auto-send exists.** Every reply requires an owner button tap, even when `can_reply` is granted.
- **Owner-only buttons.** Callbacks are authorized against the connection's `owner_user_id`; anyone else gets `⛔`.
- **Drafts expire after 24h** (configurable). Stale buttons no-op.
- **Without `can_reply`, the number buttons are hidden** — you're told to copy/paste manually instead of getting a button that silently fails.
- **Typing bursts coalesce** — messages within the debounce window accumulate into one draft (an edit replaces its part), one draft per coherent thought.
- **Only the other party is drafted.** Telegram delivers both sides of a Business chat: the owner's own messages and the bot's Send/Edit echoes are logged as context but never drafted. When the owner answers by hand, the chat's queued/pending draft is withdrawn so a stale Send can't double-reply. Messages from other bots (`from_user.is_bot`, incl. this bot's own DMs to the owner) are neither logged nor drafted.

## Requirements

- Hermes Agent with plugin Telegram-handler support (`ctx.register_telegram_handler`, hermes-agent PR #59159 / v0.18+)
- A Telegram **Business** subscription on your personal account
- Your Hermes Telegram bot with **Business Mode** toggled on in [@BotFather](https://t.me/BotFather)
- python-telegram-bot ≥ 21.1 (already installed with the Hermes Telegram platform)

## Install

This checkout is branch `local/owner-messages` (upstream `main` + owner/bot filtering, the chat log, edits, `settings` config). A fresh upstream clone lacks them; `hermes plugins update` would drop them.

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
| `/biz off <chat_id>` / `/biz on <chat_id>` | per-customer-chat mute: no drafts, no logging until `/biz on`; `off` purges the chat's log |

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

Plugin-owned SQLite at `~/.hermes/telegram-business/state.db` (three tables: connections, drafts, and `business_chat_log` — the 20 newest messages of each chat, both sides, used as draft context; purged on `/biz off <chat_id>` and when the connection ends). Hermes' core state is never touched. To reset: stop the gateway, delete the file, start it (the running process keeps the open file).

## v1 limits

- **No vision** — media enters the context as `[photo]`, `[voice message]`, ... placeholders; captions are drafted like text.
- **History starts at connection time** — bots can't read older messages; the draft sees the last 20 messages logged since.
- **No persona learning from edits** — your overrides go to the customer but don't train future drafts.

## Tests

```bash
python3 -m pytest
```

65 tests covering connection lifecycle, owner/bot message handling, the chat log, edits, config, draft supersession, debounce coalescing, all three button paths, edit capture, owner-scoped callback authorization, and `/biz` subcommands. No network, no live Telegram.

## Credits

Based on the design and implementation from [hermes-agent#30055](https://github.com/NousResearch/hermes-agent/pull/30055), replatformed as a standalone plugin. Related earlier community proposals: [#26654](https://github.com/NousResearch/hermes-agent/pull/26654) by @evgyur (earliest submission), [#35342](https://github.com/NousResearch/hermes-agent/pull/35342) by @MilekhinAV, [#46728](https://github.com/NousResearch/hermes-agent/pull/46728) by @kxnkxv.

## License

MIT
