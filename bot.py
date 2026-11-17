import os
import asyncio
import random
import uuid
import logging
import httpx
from pydub import AudioSegment
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from gemini_extension import get_gemini_reply_result, MODEL_DEFAULT, is_authorized_owner
from gemini_memory import read_history, record_exchange, clear_history
from gemini_advanced import quota_reserve, run_advanced, split_telegram_text, MAX_INPUT_CHARS
from voice_files import cleanup_voice_tempfiles

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
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TRANSIENT = {408, 429, 500, 502, 503, 504, 524, 529}


def validate_environment():
    required = {
        "TELEGRAM_TOKEN": TOKEN,
        "RENDER_EXTERNAL_URL": RENDER_URL,
        "GROQ_API_KEY": GROQ_API_KEY,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    }
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
        reply = await call_provider(
            client=client, url=GROQ_URL, key=GROQ_API_KEY,
            payload={"model": "openai/gpt-oss-120b", "messages": messages,
                     "temperature": 0.6, "max_completion_tokens": 800,
                     "reasoning_effort": "low"},
            provider_name="Groq/GPT-OSS-120B", timeout_seconds=10.0, retry_once=True,
        )
        if reply:
            return reply

        reply = await call_provider(
            client=client, url=GROQ_URL, key=GROQ_API_KEY,
            payload={"model": "qwen/qwen3.8-27b", "messages": messages,
                     "temperature": 0.6, "max_completion_tokens": 800},
            provider_name="Groq/Qwen3.8-27B", timeout_seconds=10.0, retry_once=True,
        )
        if reply:
            return reply

        reply = await call_provider(
            client=client, url=OPENROUTER_URL, key=OPENROUTER_API_KEY,
            payload={"models": ["meta-llama/llama-3.3-70b-instruct:free"],
                     "messages": messages, "temperature": 0.6, "max_tokens": 800},
            provider_name="OpenRouter", timeout_seconds=20.0, retry_once=True,
        )
        if reply:
            return reply

    return FINAL_ERROR_MESSAGE


async def transcribe_voice(file_path: str):
    try:
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "audio/wav")}
                data = {"model": "whisper-large-v3-turbo", "language": "ms"}
                response = await client.post(
                    GROQ_WHISPER_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files=files,
                    data=data,
                    timeout=30.0,
                )
            if response.status_code == 200:
                return response.json().get("text")
            log.error("Whisper error: HTTP %s - %s", response.status_code, response.text[:200])
    except Exception:
        log.exception("Gagal transkripsi voice note")
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Exclusive personal bot: silently ignore non-owner users and group chats."""
    if not update.effective_message or not is_private_owner_update(update):
        return
    await update.effective_message.reply_text(
        "Hai! Saya pembantu peribadi anda. Hantar mesej teks atau voice note ya."
    )


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the configured owner's ID only to that owner in private chat."""
    if not update.effective_message or not is_private_owner_update(update):
        return
    await update.effective_message.reply_text(
        f"Telegram user ID anda: {update.effective_user.id}"
    )


async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only Gemini chat; serialize per-chat messages to preserve order."""
    msg = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if not msg or not user or not chat:
        return
    owner_id = os.environ.get("GEMINI_OWNER_ID", "").strip()
    if not is_authorized_owner(user.id, owner_id, chat.type):
        return
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        await msg.reply_text("Gemini API key belum disediakan dalam Render.")
        return
    question = " ".join(context.args or []).strip()
    if not question:
        await msg.reply_text("Contoh: /gemini Apakah perkataan yang saya berikan tadi?")
        return
    if len(question) > 3500:
        await msg.reply_text("Soalan terlalu panjang. Had maksimum ialah 3,500 aksara.")
        return
    # PTB chat_data lives only in this process (no PicklePersistence).
    # A lock guards history when concurrent_updates is enabled in future.
    state = context.chat_data
    lock = state.setdefault("_gemini_lock_v1", asyncio.Lock())
    async with lock:
        history = read_history(state)
        await context.bot.send_chat_action(chat_id=chat.id, action="typing")
        try:
            response, ok = await asyncio.wait_for(
                get_gemini_reply_result(
                    question, api_key,
                    os.environ.get("GEMINI_MODEL", MODEL_DEFAULT),
                    history=history,
                ),
                timeout=25,
            )
        except (asyncio.TimeoutError, ValueError):
            response, ok = "Gemini gagal atau tamat masa. Semak model dan cuba lagi.", False
        except Exception:
            log.exception("Gemini handler failure")
            response, ok = "Ralat Gemini. Sila cuba lagi.", False
        if ok:
            # Save only successful exchanges, no token/API key retained.
            record_exchange(state, question, response)
        await msg.reply_text(response[:4000])


async def advanced_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only deep research or advanced analysis; explicit opt-in, zero auto-fallback."""
    msg, user, chat = update.effective_message, update.effective_user, update.effective_chat
    if not msg or not user or not chat:
        return
    if not is_authorized_owner(user.id, os.environ.get("GEMINI_OWNER_ID", ""), chat.type):
        return
    mode = (msg.text or "").split(maxsplit=1)[0].split("@")[0].lstrip("/").lower()
    if mode not in ("power", "deepdive"):
        return
    question = " ".join(context.args or []).strip()
    if not question:
        await msg.reply_text(
            "Contoh: /power Audit kod Python secara mendalam"
            if mode == "power"
            else "Contoh: /deepdive Semak perkembangan terkini, bandingkan 3 sumber rasmi"
        )
        return
    if len(question) > MAX_INPUT_CHARS:
        await msg.reply_text(f"Soalan terlalu panjang. Had maksimum {MAX_INPUT_CHARS} aksara.")
        return
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        await msg.reply_text("Gemini API key tidak tersedia dalam Render.")
        return
    state = context.chat_data
    lock = state.setdefault("_gemini_lock_v1", asyncio.Lock())
    async with lock:
        allowed, reason = quota_reserve(state, mode)
        if not allowed:
            await msg.reply_text(reason)
            return
        history = read_history(state)
        await context.bot.send_chat_action(chat_id=chat.id, action="typing")
        try:
            answer, succeeded = await asyncio.wait_for(
                run_advanced(question, api_key, mode, history=history), timeout=58,
            )
        except (asyncio.TimeoutError, ValueError):
            answer, succeeded = "Penyelidikan terlalu lama atau input tidak sah.", False
        except Exception:
            log.exception("Advanced Gemini request failed")
            answer, succeeded = "Mod Gemini tergendala; semak log Render.", False
        if succeeded:
            # Shared short-term memory across /gemini, /power, /deepdive.
            # Keep source URLs only in Telegram reply; history still capped by gemini_memory.
            record_exchange(state, question, answer)
        for part in split_telegram_text(answer):
            await msg.reply_text(part)


async def resetmemory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner can erase volatile Gemini history, within a private chat only."""
    msg, user, chat = update.effective_message, update.effective_user, update.effective_chat
    if not msg or not user or not chat:
        return
    if not is_authorized_owner(user.id, os.environ.get("GEMINI_OWNER_ID", ""), chat.type):
        return
    state = context.chat_data
    lock = state.setdefault("_gemini_lock_v1", asyncio.Lock())
    async with lock:
        clear_history(state)
        await msg.reply_text("Memori sementara Gemini telah dikosongkan.")


def is_private_owner_update(update: Update) -> bool:
    """Fail closed before any paid/free API call or Telegram media download."""
    user, chat = update.effective_user, update.effective_chat
    if not user or not chat:
        return False
    return is_authorized_owner(
        user.id, os.environ.get("GEMINI_OWNER_ID", ""), chat.type
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text or not is_private_owner_update(update):
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg.text},
    ]
    try:
        reply = await asyncio.wait_for(ask_ai(messages), timeout=55)
    except asyncio.TimeoutError:
        reply = "Maaf, respons mengambil masa terlalu lama. Cuba lagi sebentar ya."
    except Exception:
        log.exception("Unhandled error semasa ask_ai")
        reply = FINAL_ERROR_MESSAGE
    await msg.reply_text(reply[:4000])


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.voice or not is_private_owner_update(update):
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    # Per-request random names prevent concurrent requests sharing an audio path.
    nonce = uuid.uuid4().hex
    ogg_path = f"/tmp/chal_voice_{nonce}.oga"
    wav_path = f"/tmp/chal_voice_{nonce}.wav"

    # Remove audio artifacts on success, failure, and early return.
    # Never log a voice transcript or a user-linked temporary file path.
    try:
        try:
            file = await context.bot.get_file(msg.voice.file_id)
            await file.download_to_drive(ogg_path)
            audio = AudioSegment.from_file(ogg_path, format="ogg")
            audio.export(wav_path, format="wav")
        except Exception:
            log.exception("Gagal muat turun atau tukar format audio")
            await msg.reply_text("Maaf, saya tidak dapat memproses fail audio ini.")
            return

        transcript = await transcribe_voice(wav_path)
        if not transcript:
            await msg.reply_text("Maaf, saya tidak dapat memahami voice note ini. Cuba lagi.")
            return

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        try:
            reply = await asyncio.wait_for(ask_ai(messages), timeout=55)
        except asyncio.TimeoutError:
            reply = "Maaf, respons mengambil masa terlalu lama. Cuba lagi sebentar ya."
        except Exception:
            log.exception("Ralat semasa ask_ai (voice)")
            reply = FINAL_ERROR_MESSAGE

        await msg.reply_text(reply[:4000])
    finally:
        cleanup_voice_tempfiles(ogg_path, wav_path)


def main():
    validate_environment()
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("myid", myid))
    application.add_handler(CommandHandler("gemini", gemini_command))
    application.add_handler(CommandHandler("resetmemory", resetmemory_command))
    application.add_handler(CommandHandler("power", advanced_command))
    application.add_handler(CommandHandler("deepdive", advanced_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    webhook_url = f"{RENDER_URL.rstrip('/')}/{TOKEN}"
    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        url_path=TOKEN,
        webhook_url=webhook_url,
    )


if __name__ == "__main__":
    main()
