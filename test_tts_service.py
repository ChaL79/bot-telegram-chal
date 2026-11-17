"""Offline checks for owner-only Malay speech, length bounds, and safe fallback."""
import ast
import asyncio
import io
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from tts_service import (
    DEFAULT_VOICE, VOICE_CHOICES, MAX_SPEECH_CHARS,
    prepare_speech_text, normalize_malay_pronunciation,
    speech_enabled, synthesize_malay_speech,
)


class TtsTests(unittest.IsolatedAsyncioTestCase):
    def test_modes_and_cleaning(self):
        self.assertTrue(speech_enabled("auto", is_voice_note=True))
        self.assertFalse(speech_enabled("auto", is_voice_note=False))
        self.assertTrue(speech_enabled("all", is_voice_note=False))
        self.assertFalse(speech_enabled("off", is_voice_note=True))
        self.assertFalse(speech_enabled("invalid", is_voice_note=False))
        cleaned = prepare_speech_text("**Hello** [site](https://x.test/a)")
        self.assertIn("Hello", cleaned)
        self.assertIn("site", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertLessEqual(len(prepare_speech_text("hello " * 2000)),
                             MAX_SPEECH_CHARS + 65)
        self.assertEqual(prepare_speech_text(""), "")

    def test_natural_malay_spoken_decades_and_bot_name(self):
        spoken = prepare_speech_text(
            "Saya ChaL79Bot, pembantu AI peribadi. "
            "Saya suka muzik rock era 70–80."
        )
        self.assertIn("Chal tujuh puluh sembilan bot", spoken)
        self.assertIn("pembantu kecerdasan buatan", spoken)
        self.assertIn("era tujuh puluhan dan lapan puluhan", spoken)
        self.assertNotIn("70–80", spoken)
        self.assertEqual(
            normalize_malay_pronunciation("zaman 80-an"),
            "zaman lapan puluhan",
        )
        self.assertEqual(
            normalize_malay_pronunciation("era 1970-an"),
            "era tujuh puluhan",
        )

    def test_important_numbers_stay_unchanged(self):
        samples = [
            "Harga BND 70–80, bukan 90.",
            "LDT Muara 13 Nov 2026, 14:09. Jumlah 9 unit.",
            "SL 4112 dan TP 4122.",
            "MR 2026-001031. 70-80 kg.",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(normalize_malay_pronunciation(sample), sample)
        self.assertLessEqual(
            len(prepare_speech_text("ChaL79Bot " * 350)),
            MAX_SPEECH_CHARS,
        )
        self.assertEqual(
            prepare_speech_text("Hello, " + "test " * 400).endswith(
                "Jawapan penuh ada dalam mesej teks."
            ),
            True,
        )

    async def test_synthesis_is_memory_only(self):
        captured = []
        class FakeCommunicate:
            def __init__(self, phrase, *, voice):
                captured.append((phrase, voice))
            async def stream(self):
                yield {"type": "WordBoundary", "text": "Hello"}
                yield {"type": "audio", "data": b"\xff" * 100}
        mp3 = await synthesize_malay_speech(
            "Apa khabar", communicator_factory=FakeCommunicate)
        self.assertEqual(mp3, b"\xff" * 100)
        self.assertEqual(captured, [("Apa khabar", DEFAULT_VOICE)])
        with self.assertRaises(ValueError):
            await synthesize_malay_speech("Hello", voice="en-US-BadNeural",
                                         communicator_factory=FakeCommunicate)

    async def test_oversize_audio_is_rejected(self):
        class Massive:
            def __init__(self, phrase, *, voice): pass
            async def stream(self):
                yield {"type": "audio", "data": b"z" * 2_000_000}
        with self.assertRaises(ValueError):
            await synthesize_malay_speech("Hi", communicator_factory=Massive)

    async def test_owner_only_telegram_spoken_reply(self):
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
        functions = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                     and n.name == "send_spoken_reply"]
        self.assertEqual(len(functions), 1)
        synthesize = AsyncMock(return_value=b"M" * 100)
        ns = {
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "is_private_owner_update": lambda u: u.effective_user.id == 42,
            "asyncio": asyncio, "io": io, "log": logging.getLogger("tts-test"),
            "synthesize_malay_speech": synthesize, "speech_enabled": speech_enabled,
            "DEFAULT_VOICE": DEFAULT_VOICE, "MODE_AUTO": "auto",
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     "bot.py", "exec"), ns)
        audio_bot = SimpleNamespace(send_voice=AsyncMock())
        def update(user):
            return SimpleNamespace(effective_user=SimpleNamespace(id=user),
                                   effective_chat=SimpleNamespace(id=42))
        ctx = SimpleNamespace(bot=audio_bot, chat_data={})
        self.assertFalse(await ns["send_spoken_reply"](update(99), ctx, "Test", force=True))
        synthesize.assert_not_awaited()
        self.assertFalse(await ns["send_spoken_reply"](update(42), ctx, "Test"))
        self.assertTrue(await ns["send_spoken_reply"](
            update(42), ctx, "Jawapan Melayu", is_voice_note=True))
        audio_bot.send_voice.assert_awaited_once()
        self.assertEqual(audio_bot.send_voice.await_args.kwargs["voice"].name,
                         "chal_reply.mp3")

    async def test_voice_controls_and_speak_only_for_owner(self):
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                 and n.name in ("voice_command", "speak_command")]
        sender = AsyncMock(return_value=True)
        ns = {
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "is_private_owner_update": lambda u: u.effective_user.id == 42,
            "VOICE_CHOICES": VOICE_CHOICES, "DEFAULT_VOICE": DEFAULT_VOICE,
            "VOICE_MODES": {"auto", "all", "off"}, "MODE_AUTO": "auto",
            "send_spoken_reply": sender,
        }
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "bot.py", "exec"), ns)
        def update(uid):
            return SimpleNamespace(
                effective_user=SimpleNamespace(id=uid),
                effective_message=SimpleNamespace(reply_text=AsyncMock()))
        ctx = SimpleNamespace(args=["all"], chat_data={})
        await ns["voice_command"](update(99), ctx)
        self.assertEqual(ctx.chat_data, {})
        await ns["voice_command"](update(42), ctx)
        self.assertEqual(ctx.chat_data["_tts_mode"], "all")
        ctx.args = ["wanita"]
        await ns["voice_command"](update(42), ctx)
        self.assertEqual(ctx.chat_data["_tts_voice"], VOICE_CHOICES["wanita"])
        ctx.args = ["Selamat", "pagi"]
        await ns["speak_command"](update(99), ctx)
        sender.assert_not_awaited()
        await ns["speak_command"](update(42), ctx)
        sender.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
