import os
import logging
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get("TELEGRAM_TOKEN")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hai! Saya pembantu peribadi anda. Apa yang boleh saya bantu?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": RENDER_URL or "https://t.me/yourbot",
                    "X-Title": "Bot Pembantu Peribadi"
                },
                json={
                    "models": [
                        "qwen/qwen3.6-plus:free",
                        "nvidia/nemotron-3-ultra-550b-a55b:free"
                    ],
                    "messages": [
                        {"role": "system", "content": "Anda pembantu peribadi AI yang mesra. Jawab dalam Bahasa Melayu."},
                        {"role": "user", "content": update.message.text}
                    ]
                }
            )
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = f"Maaf, ada masalah teknikal. (Error: {response.status_code})"
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text(f"Maaf, ada masalah teknikal. (Error: {e})")

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        url_path=TOKEN,
        webhook_url=f"{RENDER_URL}/{TOKEN}",
    )

if __name__ == "__main__":
    main()
