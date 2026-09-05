"""
Standalone Telegram adapter for Sarah.

Runs as its own long-lived process, separate from Sarah_Core. Normalizes
incoming Telegram messages into Sarah's {platform, platform_user_id, text}
shape, POSTs to /turn, and relays the text response back.

Setup:
    pip install python-telegram-bot httpx
    export TELEGRAM_BOT_TOKEN=<token from @BotFather>
    python telegram_adapter.py
"""

import logging
import os

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("telegram_adapter")

SARAH_URL = os.environ.get("SARAH_URL", "http://localhost:9000/turn")
BOT_TOKEN = "put your own token"  # required - fail loud at startup, not on first message

# Local 4B generation plus a possible tool-call loop isn't instant - give it
# real headroom rather than hitting httpx's own default timeout mid-turn.
REQUEST_TIMEOUT_S = 60.0


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = str(update.effective_chat.id)
    text = update.message.text

    # Telegram's "typing..." indicator - closest equivalent to the
    # "thinking"/"speaking" status events the WS client gets.
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    payload = {
        "platform": "telegram",
        "platform_user_id": chat_id,
        "text": text,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            resp = await client.post(SARAH_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        log.warning("Sarah_Core timed out for chat %s", chat_id)
        await update.message.reply_text("Sarah's still thinking - give it another moment and try again.")
        return
    except httpx.HTTPStatusError as e:
        log.error("Sarah_Core returned %s for chat %s: %s", e.response.status_code, chat_id, e.response.text)
        await update.message.reply_text("Something went wrong on Sarah's end.")
        return
    except Exception as e:
        log.error("Failed to reach Sarah_Core: %s", e)
        await update.message.reply_text("Couldn't reach Sarah right now.")
        return

    reply_text = data.get("text", "").strip()
    if not reply_text:
        await update.message.reply_text("(no response)")
        return

    await update.message.reply_text(reply_text)


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Telegram adapter starting, forwarding to %s", SARAH_URL)
    app.run_polling()


if __name__ == "__main__":
    main()