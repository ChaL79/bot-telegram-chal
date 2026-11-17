"""Opt-in, owner-only pinned Telegram note memory.

This intentionally does not use Render's ephemeral disk or a paid database.
Only explicit /remember input is retained as an editable Telegram message.
It is NOT an encrypted vault or a general conversation archive.
"""
import re

MARKER = "CHAL79_PRIVATE_MEMORY_V1"
MAX_NOTES = 12
MAX_NOTE_CHARS = 180
MAX_DISPLAY_CHARS = 3200

FORBIDDEN = re.compile(
    r"(?i)(password|kata\s*laluan|passphrase|api[ _-]?key|access[ _-]?token|"
    r"secret|otp|private[ _-]?key|cvv|pin\s*(?:bank|kad|card)|"
    r"seed[ _-]?phrase|recovery[ _-]?phrase)"
)


def validate_new_note(note: str) -> str:
    note = " ".join(note.strip().split())
    if not note or len(note) > MAX_NOTE_CHARS:
        raise ValueError("Nota mesti 1 hingga 180 aksara.")
    if "\n" in note or FORBIDDEN.search(note):
        raise ValueError("Jangan simpan kata laluan, token API atau maklumat rahsia.")
    return note


def serialize_notes(notes: list[str]) -> str:
    if len(notes) > MAX_NOTES or any(
        not isinstance(v, str) or not v or len(v) > MAX_NOTE_CHARS
        for v in notes
    ):
        raise ValueError("Memori melebihi had.")
    body = "\n".join(f"{i}. {v}" for i, v in enumerate(notes, 1))
    text = MARKER + "\n" + body
    if len(text) > MAX_DISPLAY_CHARS:
        raise ValueError("Memori terlalu panjang.")
    return text


def parse_notes(content: str | None) -> list[str] | None:
    if not isinstance(content, str) or not content.startswith(MARKER + "\n"):
        return None
    if len(content) > MAX_DISPLAY_CHARS:
        return None
    rest = content.split("\n")[1:]
    if not rest or rest == [""]:
        return []
    if len(rest) > MAX_NOTES:
        return None
    result = []
    for index, line in enumerate(rest, 1):
        prefix = f"{index}. "
        if not line.startswith(prefix):
            return None
        note = line[len(prefix):]
        if not note or len(note) > MAX_NOTE_CHARS:
            return None
        result.append(note)
    return result


async def get_pinned_notes(bot, chat_id: int) -> tuple[list[str] | None, int | None]:
    """Return (None,None) when pinned message isn't ours; never overwrite it."""
    chat = await bot.get_chat(chat_id)
    message = getattr(chat, "pinned_message", None)
    if message is None:
        return [], None
    notes = parse_notes(getattr(message, "text", None))
    if notes is None:
        return None, None
    return notes, message.message_id


async def write_pinned_notes(bot, chat_id: int, notes: list[str], message_id: int | None) -> None:
    value = serialize_notes(notes)
    if message_id is None:
        sent = await bot.send_message(chat_id=chat_id, text=value)
        await bot.pin_chat_message(
            chat_id=chat_id, message_id=sent.message_id,
            disable_notification=True)
    else:
        await bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=value)
