"""Owner-only, opt-in Gemini integration. Existing Groq/OpenRouter routing is untouched."""
import logging
import re
from typing import Optional
import httpx

log = logging.getLogger("gemini_extension")
MODEL_DEFAULT = "gemini-3.5-flash-lite"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _safe_model(model: str) -> str:
    """Validate the model path segment before building the URL."""
    if not isinstance(model, str) or not re.fullmatch(r"gemini-[A-Za-z0-9_.-]+", model):
        raise ValueError("GEMINI_MODEL tidak sah")
    return model


def is_authorized_owner(user_id: Optional[int], owner_id: str, chat_type: Optional[str]) -> bool:
    """Deny API access if owner ID is missing, invalid or chat is not private."""
    return (
        isinstance(user_id, int)
        and not isinstance(user_id, bool)
        and chat_type == "private"
        and bool(owner_id and owner_id.strip().isdigit())
        and str(user_id) == owner_id.strip()
    )


def extract_gemini_text(data: dict) -> Optional[str]:
    """Extract visible text only, skipping thought parts."""
    if not isinstance(data, dict):
        return None
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    output = "\n".join(
        p["text"] for p in parts
        if isinstance(p, dict)
        and not p.get("thought", False)
        and isinstance(p.get("text"), str)
    )
    return output.strip() or None


async def get_gemini_reply(
    text: str, api_key: str, model: str = MODEL_DEFAULT, *, client: Optional[httpx.AsyncClient] = None
) -> str:
    """Single bounded request, no retry, no automatic fallback or spending."""
    if not api_key:
        return "Gemini API key belum dikonfigurasikan."
    if not text or not text.strip():
        return "Taip /gemini diikuti soalan anda."
    if len(text) > 3500:
        return "Soalan terlalu panjang. Had maksimum ialah 3,500 aksara."
    model = _safe_model(model)
    url = f"{GEMINI_BASE}/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": "Anda pembantu AI yang mesra. Jawab dalam Bahasa Melayu."}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 650},
    }

    async def send(session: httpx.AsyncClient) -> str:
        try:
            response = await session.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(20.0, connect=5.0),
            )
        except (httpx.TimeoutException, httpx.RequestError):
            log.warning("Gemini network/timeout failure")
            return "Gemini tidak dapat dihubungi sekarang. Cuba lagi kemudian."

        if response.status_code == 429:
            log.info("Gemini rate limit reached")
            return "Had penggunaan Gemini dicapai. Cuba lagi selepas kuota reset."
        if response.status_code in (401, 403):
            log.warning("Gemini authorization error (%d)", response.status_code)
            return "Kunci Gemini tidak mempunyai akses. Semak projek Free Tier dan API key."
        if response.status_code != 200:
            log.warning("Gemini HTTP error (%d)", response.status_code)
            return "Gemini tidak berjaya menjawab. Semak model/kuota dalam AI Studio."
        try:
            answer = extract_gemini_text(response.json())
        except ValueError:
            answer = None
        return (answer or "Gemini tidak mengembalikan jawapan teks.")[:4000]

    if client is not None:
        return await send(client)
    async with httpx.AsyncClient() as new_client:
        return await send(new_client)
