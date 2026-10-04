import os
import threading
import requests
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot sedang berjalan!"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hai! Saya pembantu peribadi anda. Apa yang boleh saya bantu?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {"role": "system", "content": "Anda pembantu peribadi AI yang mesra. Jawab dalam Bahasa Melayu."},
                    {"role": "user", "content": user_message}
                ]
            }
        )
        
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
        else:
            reply = f"Maaf, ada masalah teknikal. (Error: {response.status_code})"
            
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)
        
    except Exception as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Maaf, ada masalah teknikal. (Error: {e})")

def run_bot():
    application = ApplicationBuilder().token(os.environ.get("TELEGRAM_TOKEN")).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.run_polling()

if __name__ == '__main__':
    threading.Thread(target=run_bot).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)