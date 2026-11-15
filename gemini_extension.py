"""Owner-only, opt-in Gemini integration. Existing Groq/OpenRouter routing is untouched."""
import logging
import re
from typing import Optional, Sequence
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


async def get_gemini_reply_result(
    text: str, api_key: str, model: str = MODEL_DEFAULT, *,
    history: Optional[Sequence[dict]] = None,
    client: Optional[httpx.AsyncClient] = None
) -> tuple[str, bool]:
    """Return (visible reply, succeeded). Only successful API results enter history."""
    if not api_key:
        return "Gemini API key belum dikonfigurasikan.", False
    if not text or not text.strip():
        return "Taip /gemini diikuti soalan anda.", False
    if len(text) > 3500:
        return "Soalan terlalu panjang. Had maksimum ialah 3,500 aksara.", False
    model = _safe_model(model)
    url = f"{GEMINI_BASE}/{model}:generateContent"
    # Caller provides only bounded, vetted, complete user/model pairs.
    # A new user message must always be last in the Gemini contents array.
    validated = []
    if history:
        if len(history) > 10 or len(history) % 2:
            raise ValueError("Gemini history length invalid")
        for i, entry in enumerate(history):
            role = "user" if i % 2 == 0 else "model"
            if not isinstance(entry, dict) or entry.get("role") != role:
                raise ValueError("Gemini history role invalid")
            parts = entry.get("parts")
            if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
                raise ValueError("Gemini history format invalid")
            val = parts[0].get("text")
            if not isinstance(val, str) or len(val) > 1400:
                raise ValueError("Gemini history text invalid")
            validated.append({"role": role, "parts": [{"text": val}]})
    payload = {
        "systemInstruction": {"parts": [{"text": "Anda pembantu AI yang mesra. Jawab dalam Bahasa Melayu."}]},
        "contents": validated + [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 650},
    }

    async def send(session: httpx.AsyncClient) -> tuple[str, bool]:
        try:
            response = await session.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(20.0, connect=5.0),
            )
        except (httpx.TimeoutException, httpx.RequestError):
            log.warning("Gemini network/timeout failure")
            return "Gemini tidak dapat dihubungi sekarang. Cuba lagi kemudian.", False

        if response.status_code == 429:
            log.info("Gemini rate limit reached")
            return "Had penggunaan Gemini dicapai. Cuba lagi selepas kuota reset.", False
        if response.status_code in (401, 403):
            log.warning("Gemini authorization error (%d)", response.status_code)
            return "Kunci Gemini tidak mempunyai akses. Semak projek Free Tier dan API key.", False
        if response.status_code != 200:
            log.warning("Gemini HTTP error (%d)", response.status_code)
            return "Gemini tidak berjaya menjawab. Semak model/kuota dalam AI Studio.", False
        try:
            answer = extract_gemini_text(response.json())
        except ValueError:
            answer = None
        if not answer:
            return "Gemini tidak mengembalikan jawapan teks.", False
        return answer[:4000], True

    if client is not None:
        return await send(client)
    async with httpx.AsyncClient() as new_client:
        return await send(new_client)


async def get_gemini_reply(
    text: str, api_key: str, model: str = MODEL_DEFAULT, *,
    history: Optional[Sequence[dict]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Backward-compatible interface used by the existing API tests."""
    reply, _success = await get_gemini_reply_result(
        text, api_key, model, history=history, client=client
    )
    return reply
