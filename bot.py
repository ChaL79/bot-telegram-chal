import os
import asyncio
import random
import logging
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get("TELEGRAM_TOKEN")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("bot")

SYSTEM_PROMPT = "Anda pembantu peribadi AI yang mesra. Jawab dalam Bahasa Melayu."
FINAL_ERROR_MESSAGE = "Maaf, servis AI sedang sibuk sekarang. Cuba hantar mesej sekali lagi sebentar lagi ya."
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TRANSIENT = {408, 429, 500, 502, 503, 504, 524, 529}

def validate_environment():
    required = {"TELEGRAM_TOKEN": TOKEN, "RENDER_EXTERNAL_URL": RENDER_URL,
                "GROQ_API_KEY": GROQ_API_KEY, "OPENROUTER_API_KEY": OPENROUTER_API_KEY}
    missing = [n for n, v in required.items() if not v]
    if missing:
        raise RuntimeError("Environment variable tiada: " + ", ".join(missing))

def get_retry_after(response):
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None

def extract_content(data):
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    content = content.strip()
    return content if content else None

async def call_provider(client, url, key, payload, provider_name,
                        timeout_seconds=10.0, retry_once=True):
    max_attempts = 2 if retry_once else 1
    for attempt in range(max_attempts):
        try:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(timeout_seconds, connect=5.0),
            )
        except httpx.TimeoutException:
            log.warning("%s: request timeout", provider_name)
            return None
        except httpx.RequestError as exc:
            log.warning("%s: network error: %s", provider_name, type(exc).__name__)
            return None

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                log.warning("%s: HTTP 200 tetapi JSON tidak sah", provider_name)
                return None
            text = extract_content(data)
            if text:
                log.info("%s: berjaya | model=%s", provider_name, data.get("model", "unknown"))
                return text
            api_error = data.get("error")
            if api_error:
                log.warning("%s: API response error: %s", provider_name, str(api_error)[:300])
            else:
                log.warning("%s: respons 200 tetapi tiada assistant content", provider_name)
            return None

        if response.status_code not in TRANSIENT:
            log.error("%s: HTTP %s - %s", provider_name, response.status_code, response.text[:300])
            return None

        log.warning("%s: transient HTTP %s", provider_name, response.status_code)
        if attempt >= max_attempts - 1:
            return None

        retry_after = get_retry_after(response)
        if retry_after is not None:
            if retry_after > 2.0:
                log.warning("%s: Retry-After %.1fs terlalu lama; terus fallback", provider_name, retry_after)
                return None
            delay = retry_after
        else:
            delay = 0.8 * (2 ** attempt)
        delay += random.uniform(0.0, 0.3)
        log.info("%s: retry sekali dalam %.2fs", provider_name, delay)
        await asyncio.sleep(delay)
    return None

async def ask_ai(messages):
    async with httpx.AsyncClient() as client:
        # 1. Groq GPT-OSS 120B
        reply = await call_provider(
            client=client, url=GROQ_URL, key=GROQ_API_KEY,
            payload={"model": "openai/gpt-oss-120b", "messages": messages,
                     "temperature": 0.6, "max_completion_tokens": 800,
                     "reasoning_effort": "low"},
            provider_name="Groq/GPT-OSS-120B", timeout_seconds=10.0, retry_once=True,
        )
        if reply: return reply

        # 2. Groq Qwen 3.8 27B
        reply = await call_provider(
            client=client, url=GROQ_URL, key=GROQ_API_KEY,
            payload={"model": "qwen/qwen3.8-27b", "messages": messages,
                     "temperature": 0.6, "max_completion_tokens": 800},
            provider_name="Groq/Qwen3.8-27B", timeout_seconds=10.0, retry_once=True,
        )
        if reply: return reply

        # 3 & 4. OpenRouter (native fallback)
        reply = await call_provider(
            client=client, url=OPENROUTER_URL, key=OPENROUTER_API_KEY,
            payload={"models": ["qwen/qwen3.6-plus:free",
                                "meta-llama/llama-3.3-70b-instruct:free"],
                     "messages": messages, "temperature": 0.6, "max_tokens": 800},
            provider_name="OpenRouter", timeout_seconds=20.0, retry_once=True,
        )
        if reply: return reply

    return FINAL_ERROR_MESSAGE

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Hai! Saya pembantu peribadi anda. Apa yang boleh saya bantu?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text:
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg.text},
    ]
    try:
        reply = await asyncio.wait_for(ask_ai(messages), timeout=55)
    except asyncio.TimeoutError:
        log.warning("Keseluruhan AI fallback chain timeout")
        reply = "Maaf, respons mengambil masa terlalu lama. Cuba lagi sebentar ya."
    except Exception:
        log.exception("Unhandled error semasa ask_ai")
        reply = FINAL_ERROR_MESSAGE
    await msg.reply_text(reply[:4000])

def main():
    validate_environment()
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    webhook_url = f"{RENDER_URL.rstrip('/')}/{TOKEN}"
    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        url_path=TOKEN,
        webhook_url=webhook_url,
    )

if __name__ == "__main__":
    main()
