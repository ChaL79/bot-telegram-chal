"""Personal Malay TTS using an opt-in/auto voice reply path.

edge-tts uses an unofficial Microsoft Edge online speech service. It does not
need a paid API key, but availability is not guaranteed. Only the private owner
can invoke this module through the Telegram handlers. User text is sent to
Microsoft for synthesis; audio is held in RAM only and never logged.
"""
import asyncio
import re

VOICE_MALE = "ms-MY-OsmanNeural"
VOICE_FEMALE = "ms-MY-YasminNeural"
VOICE_CHOICES = {"lelaki": VOICE_MALE, "wanita": VOICE_FEMALE}
DEFAULT_VOICE = VOICE_MALE

MAX_SPEECH_CHARS = 950
MAX_SPEECH_BYTES = 1_750_000
TTS_TIMEOUT_SECONDS = 27

# Modes are process-memory preferences. New Render process defaults to voice
# replies for voice notes, text-only for typed messages.
MODE_AUTO = "auto"
MODE_ALL = "all"
MODE_OFF = "off"
VOICE_MODES = frozenset({MODE_AUTO, MODE_ALL, MODE_OFF})


def speech_enabled(mode: str, *, is_voice_note: bool) -> bool:
    if mode not in VOICE_MODES:
        mode = MODE_AUTO
    return mode == MODE_ALL or (mode == MODE_AUTO and is_voice_note)


def prepare_speech_text(answer: str) -> str:
    """Read meaningful plain text, never code fences, URLs or raw markdown."""
    if not isinstance(answer, str) or not answer.strip():
        return ""
    plain = re.sub(r"(?s)```.*?```", " [blok kod tidak dibaca] ", answer)
    plain = re.sub(r"https?://[^\s)]+", "pautan dalam teks", plain)
    plain = re.sub(r"\[([^\]]{1,100})\]\([^)]+\)", r"\1", plain)
    plain = re.sub(r"[#*_~\`]+", "", plain)
    plain = re.sub(r"[\u200b-\u200d\ufeff]", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) <= MAX_SPEECH_CHARS:
        return plain
    trimmed = plain[:MAX_SPEECH_CHARS].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return trimmed + ". Jawapan penuh ada dalam mesej teks."


async def synthesize_malay_speech(
    text: str, *, voice: str = DEFAULT_VOICE, communicator_factory=None
) -> bytes:
    """Return a bounded MP3 in RAM. Raise ValueError on empty/oversized audio.

    communicator_factory is injected only in offline tests. The network service
    is not an API with guaranteed service levels.
    """
    if voice not in VOICE_CHOICES.values():
        raise ValueError("Unsupported TTS voice")
    prepared = prepare_speech_text(text)
    if not prepared:
        raise ValueError("Empty speech")
    if communicator_factory is None:
        import edge_tts
        communicator_factory = edge_tts.Communicate

    async def collect() -> bytes:
        chunks = bytearray()
        communicator = communicator_factory(prepared, voice=voice)
        async for event in communicator.stream():
            if not isinstance(event, dict) or event.get("type") != "audio":
                continue
            data = event.get("data")
            if not isinstance(data, (bytes, bytearray)):
                continue
            if len(chunks) + len(data) > MAX_SPEECH_BYTES:
                raise ValueError("TTS output too large")
            chunks.extend(data)
        if len(chunks) < 50:
            raise ValueError("Empty TTS audio")
        return bytes(chunks)

    return await asyncio.wait_for(collect(), timeout=TTS_TIMEOUT_SECONDS)
