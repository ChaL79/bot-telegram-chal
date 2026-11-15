"""Bounded, temporary Gemini chat memory stored in PTB chat_data (RAM only)."""
import time
from typing import MutableMapping, Optional

HISTORY_KEY = "_gemini_history_v1"
LAST_ACTIVE_KEY = "_gemini_last_success_v1"
MAX_MESSAGES = 10  # Five user/model exchanges, including both sides
MAX_CHARS_PER_MESSAGE = 1400
MAX_HISTORY_CHARS = 10000
IDLE_TTL_SECONDS = 60 * 60


def clear_history(chat_data: MutableMapping) -> bool:
    """Discard the conversation without persisting its contents anywhere."""
    had_history = bool(chat_data.pop(HISTORY_KEY, []))
    chat_data.pop(LAST_ACTIVE_KEY, None)
    return had_history


def read_history(chat_data: MutableMapping, *, now: Optional[float] = None) -> list:
    """Return an independent copy; expire history after an hour of inactivity."""
    when = time.monotonic() if now is None else now
    stamp = chat_data.get(LAST_ACTIVE_KEY)
    if not isinstance(stamp, (int, float)) or when - stamp >= IDLE_TTL_SECONDS or when < stamp:
        clear_history(chat_data)
        return []
    entries = chat_data.get(HISTORY_KEY, [])
    if not isinstance(entries, list):
        clear_history(chat_data)
        return []
    # No mutable references to chat_data escape to the API caller.
    copy = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in ("user", "model"):
            clear_history(chat_data)
            return []
        parts = entry.get("parts")
        if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
            clear_history(chat_data)
            return []
        value = parts[0].get("text")
        if not isinstance(value, str):
            clear_history(chat_data)
            return []
        copy.append({"role": entry["role"], "parts": [{"text": value[:MAX_CHARS_PER_MESSAGE]}]})
    # History must have complete pairs that alternate exactly.
    if len(copy) % 2 or any(item["role"] != ("user" if n % 2 == 0 else "model") for n, item in enumerate(copy)):
        clear_history(chat_data)
        return []
    return copy[-MAX_MESSAGES:]


def record_exchange(chat_data: MutableMapping, question: str, answer: str, *, now: Optional[float] = None) -> None:
    """Append only successful exchanges, keeping oldest complete pairs out of the budget."""
    when = time.monotonic() if now is None else now
    history = read_history(chat_data, now=when)
    history.extend([
        {"role": "user", "parts": [{"text": question[:MAX_CHARS_PER_MESSAGE]}]},
        {"role": "model", "parts": [{"text": answer[:MAX_CHARS_PER_MESSAGE]}]},
    ])
    history = history[-MAX_MESSAGES:]
    def size(items):
        return sum(len(item["parts"][0]["text"]) for item in items)
    while len(history) > 2 and size(history) > MAX_HISTORY_CHARS:
        history = history[2:]
    chat_data[HISTORY_KEY] = history
    chat_data[LAST_ACTIVE_KEY] = when
