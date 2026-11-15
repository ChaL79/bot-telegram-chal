"""Free-tier aware advanced reasoning and sourced web search for owner-only Telegram bot.

Commands:
 /power => Gemini 3.8 Flash, high reasoning, no web.
 /deepdive => Gemini 2.5 Flash, Google Search grounding with source metadata.
No paid-tier activation, retries, alternate endpoints, external browser or code execution.
"""
import datetime as dt
import logging
import re
import time
from collections import deque
from urllib.parse import urlsplit
from typing import Optional

import httpx

log = logging.getLogger("gemini_advanced")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
POWER_MODEL = "gemini-3.8-flash"
RESEARCH_MODEL = "gemini-2.5-flash"
QUOTA_KEY = "_advanced_daily_quota_v1"
MODE_LIMITS = {"power": (2, 12), "deepdive": (2, 12)}  # RPM, local per-UTC-day cap
MAX_INPUT_CHARS = 3200

RESEARCH_PROMPT = (
    "Anda penyelidik AI berintegriti. Jawab dalam Bahasa Melayu yang jelas. "
    "Cari di Google Search untuk soalan ini. Gunakan sumber terkini apabila relevan. "
    "Bezakan dengan jelas: (1) fakta yang disokong sumber, "
    "(2) analisis/inferens, (3) perkara tidak dapat disahkan. "
    "Bandingkan sumber yang bercanggah tanpa mereka fakta atau URL. "
    "Jika perlu, beri langkah praktikal. Untuk harga pasaran, jadual dan berita, "
    "nyatakan tarikh dan masa sumber; jangan mendakwa live tanpa data bertarikh. "
    "Jangan jalankan perdagangan, pembelian, mesej luaran atau tindakan pada akaun."
)
POWER_PROMPT = (
    "Anda penganalisis AI teknikal yang teliti. Jawab dalam Bahasa Melayu. "
    "Utamakan ketepatan, pengiraan yang boleh diperiksa, hujah pro-kontra, "
    "semakan andaian, risiko dan langkah praktikal. Periksa logik dalaman. "
    "Jelaskan apabila anda tidak tahu atau tiada akses Internet/data live. "
    "Jangan mereka sumber, statistik atau dapatan ujian. "
    "Jangan lakukan transaksi, perubahan akaun, deploy atau auto trading."
)


def quota_reserve(state: dict, mode: str, *, monotonic_now: Optional[float] = None,
                  utc_day: Optional[str] = None) -> tuple[bool, str]:
    """Atomic only if called under a per-chat asyncio.Lock. RAM-only, per-owner guard."""
    if mode not in MODE_LIMITS:
        raise ValueError("unsupported advanced mode")
    now = time.monotonic() if monotonic_now is None else monotonic_now
    today = dt.datetime.now(dt.timezone.utc).date().isoformat() if utc_day is None else utc_day
    rpm, daily_limit = MODE_LIMITS[mode]
    quotas = state.setdefault(QUOTA_KEY, {})
    current = quotas.get(mode)
    if not isinstance(current, dict) or current.get("utc_day") != today:
        current = {"utc_day": today, "daily": 0, "timestamps": deque()}
        quotas[mode] = current
    ts = current["timestamps"]
    # Drop stale entries; handle monotonic reset and corrupted memory conservatively.
    if not isinstance(ts, deque):
        ts = deque()
        current["timestamps"] = ts
    while ts and (ts[0] <= now - 60 or ts[0] > now):
        ts.popleft()
    if current["daily"] >= daily_limit:
        return False, f"Had dalaman {mode} hari ini telah dicapai ({daily_limit} permintaan)."
    if len(ts) >= rpm:
        return False, "Had dalaman 2 permintaan/minit dicapai. Cuba lagi selepas seminit."
    current["daily"] += 1
    ts.append(now)
    return True, ""


def safe_web_sources(data: dict, max_sources: int = 5) -> list[tuple[str, str]]:
    """Only surface real URLs returned as grounding metadata, never invented links."""
    try:
        chunks = data["candidates"][0]["groundingMetadata"]["groundingChunks"]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(chunks, list):
        return []
    result, seen = [], set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        raw_url = web.get("uri")
        if not isinstance(raw_url, str) or len(raw_url) > 750 or any(x in raw_url for x in "\n\r\t<>"):
            continue
        parsed = urlsplit(raw_url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.username or parsed.password:
            continue
        if raw_url in seen:
            continue
        seen.add(raw_url)
        title = web.get("title", parsed.hostname or "Sumber")
        title = re.sub(r"[\r\n\t<>]", " ", str(title)).strip()[:110]
        result.append((title or "Sumber", raw_url))
        if len(result) >= max_sources:
            break
    return result


def visible_reply(data: dict) -> Optional[str]:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(parts, list):
        return None
    content = "\n".join(
        part["text"] for part in parts
        if isinstance(part, dict) and not part.get("thought")
        and isinstance(part.get("text"), str)
    ).strip()
    return content or None


def split_telegram_text(message: str, size: int = 3800) -> list[str]:
    """Keep text under Telegram's 4096 limit; avoid splitting a link if possible."""
    if not message:
        return []
    chunks = []
    remainder = message
    while remainder:
        if len(remainder) <= size:
            chunks.append(remainder)
            break
        i = remainder.rfind("\n", 0, size)
        if i < size // 2:
            i = size
        chunks.append(remainder[:i])
        remainder = remainder[i:].lstrip("\n")
    return chunks


def _verified_history(history: Optional[list]) -> list[dict]:
    if not history:
        return []
    if len(history) > 10 or len(history) % 2:
        raise ValueError("History size invalid")
    out = []
    for index, msg in enumerate(history):
        expected = "user" if index % 2 == 0 else "model"
        if not isinstance(msg, dict) or msg.get("role") != expected:
            raise ValueError("History role invalid")
        parts = msg.get("parts")
        if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
            raise ValueError("History parts invalid")
        text = parts[0].get("text")
        if not isinstance(text, str) or len(text) > 1400:
            raise ValueError("History text invalid")
        out.append({"role": expected, "parts": [{"text": text}]})
    return out


async def run_advanced(
    question: str,
    api_key: str,
    mode: str,
    *,
    history: Optional[list] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> tuple[str, bool]:
    """Returns text and API-success flag; no retries. Never records unsuccessful content."""
    if mode not in MODE_LIMITS:
        raise ValueError("Invalid mode")
    if not api_key:
        return "Gemini API key tidak tersedia.", False
    if not question.strip():
        return f"Taip /{mode} diikuti soalan.", False
    if len(question) > MAX_INPUT_CHARS:
        return f"Soalan melebihi {MAX_INPUT_CHARS} aksara.", False
    history_items = _verified_history(history)
    research = mode == "deepdive"
    model = RESEARCH_MODEL if research else POWER_MODEL
    # GenerateContent is used for compatibility with existing Gemini bot.
    prompt = RESEARCH_PROMPT if research else POWER_PROMPT
    current_date = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%d %B %Y")
    payload = {
        "systemInstruction": {"parts": [{"text": prompt + " Tarikh rujukan (Brunei): " + current_date + "."}]},
        "contents": history_items + [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {
            "temperature": 0.35 if research else 0.45,
            "maxOutputTokens": 4200 if research else 7000,
            "thinkingConfig": {"thinkingBudget": 512} if research else {"thinkingLevel": "high"},
        },
    }
    if research:
        payload["tools"] = [{"google_search": {}}]

    async def request(session: httpx.AsyncClient) -> tuple[str, bool]:
        try:
            response = await session.post(
                f"{GEMINI_API_BASE}/{model}:generateContent",
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(52.0, connect=6.0),
            )
        except (httpx.TimeoutException, httpx.RequestError):
            log.warning("Advanced Gemini %s network failure", mode)
            return "Servis Gemini lambat atau tidak dapat dihubungi. Cuba lagi kemudian.", False
        if response.status_code == 429:
            return "Kuota Free Tier Gemini untuk model ini sudah dicapai. Cuba lagi kemudian.", False
        if response.status_code in (401, 403):
            return "Akses API tidak dibenarkan. Semak API key / Free Tier pada AI Studio.", False
        if response.status_code in (400, 404):
            log.warning("Unsupported Gemini model/tool mode=%s HTTP=%s", mode, response.status_code)
            return (
                "Model atau alat ini belum tersedia melalui Free Tier/projek awak. "
                "Tidak menggunakan carian berbayar atau fallback yang mengelirukan.", False
            )
        if response.status_code != 200:
            log.warning("Gemini advanced %s HTTP %s", mode, response.status_code)
            return "Gemini tidak dapat menjawab sekarang. Cuba lagi kemudian.", False
        try:
            data = response.json()
        except ValueError:
            return "Gemini memberikan respons tidak sah.", False
        answer = visible_reply(data)
        if not answer:
            return "Gemini tidak mengembalikan jawapan teks. Cuba soalan lebih ringkas.", False
        if not research:
            return answer[:10700], True
        sources = safe_web_sources(data)
        # Google Search grounding may not execute for each prompt. Do not fake evidence.
        if sources:
            references = "\n\nSumber daripada Google Search:\n" + "\n".join(
                f"{i}. {title}\n{url}" for i, (title, url) in enumerate(sources, 1)
            )
            return answer[:max(2000, 10000 - len(references))] + references, True
        return (
            answer[:9000] + "\n\n⚠️ Tiada pautan sumber daripada metadata Google Search "
            "untuk jawapan ini; dakwaan semasa belum dapat disahkan melalui carian.",
            True,
        )

    if client is not None:
        return await request(client)
    async with httpx.AsyncClient() as session:
        return await request(session)
