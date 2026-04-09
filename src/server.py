# A simple Long-polling server
from pprint import pprint
from zalo_bot import Update
from zalo_bot.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from agents import SQLiteSession
from zalo_bot.constants import ChatAction
from dotenv import load_dotenv
import os
from src.agent import NewsAgent


load_dotenv()
news_agent = NewsAgent(debug=True)

# Session ID for accessing to old previous messages (persistent memory?)
# we temporarily use a dummy session ID for quick testing.
session_id = "conversation_170296"
conversation_db = "conversation.db"
session = SQLiteSession(session_id, conversation_db) # in-memory database


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Called `start` function.")
    await update.message.reply_text(f"Hello {update.effective_user.display_name}! I am Luky")


async def reply_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # await update.message.reply_text(f"You just said: {update.message.text}")
    print("[DEBUG][server.py] The agent is thinking...")
    pprint(f"[DEBUG][server.py] Current Session: {await session.get_items(limit=5)}")

    # Typing effect
    await context.bot.send_chat_action(
        chat_id=update.message.chat.id,
        action=ChatAction.TYPING
    )
    response = await news_agent.reply(query=update.message.text, session=session)
    await update.message.reply_text(response)
    print(f"Agent responded: {response}")


async def reply_for_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = update.message.api_kwargs.get("caption", "")
    photo_url = update.message.api_kwargs.get("photo_url", "")

    print("[DEBUG][server.py] Called reply_for_photo.")
    pprint(f"[DEBUG][server.py] Received Message: {update.message}")
    print(f"[DEBUG][server.py] Caption: {caption}")
    print(f"[DEBUG][server.py] Photo URL: {photo_url}")
    pprint(f"[DEBUG][server.py] Current Session: {await session.get_items()}")

    # Typing effect
    await context.bot.send_chat_action(
        chat_id=update.message.chat.id,
        action=ChatAction.TYPING
    )
    response = await news_agent.reply(query=caption, photo_url=photo_url, session=session)
    await update.message.reply_text(response)


if __name__ == "__main__":
    app = ApplicationBuilder().token(os.getenv("ZALO_BOT_TOKEN")).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, reply_user))
    app.add_handler(MessageHandler(filters.PHOTO, reply_for_photo))

    print("Bot is running...")

    try:
        app.bot.delete_webhook()
        app.run_polling()
    except KeyboardInterrupt:
        print("Bot is killed.")
    except Exception as e:
        print(e)