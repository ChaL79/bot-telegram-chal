import unittest
from types import SimpleNamespace

from memory_notes import (
    MARKER, MAX_NOTES, validate_new_note, parse_notes,
    serialize_notes, get_pinned_notes, write_pinned_notes
)


class MemoryNotesTests(unittest.IsolatedAsyncioTestCase):
    def test_roundtrip_and_empty(self):
        notes = ["Saya suka jawapan ringkas", "Kerja offshore"]
        self.assertEqual(parse_notes(serialize_notes(notes)), notes)
        self.assertEqual(parse_notes(serialize_notes([])), [])
        self.assertIsNone(parse_notes("Another person's pinned message"))
        self.assertIsNone(parse_notes(MARKER + "\n2. Malformed"))

    def test_sensitive_or_long_data_rejected(self):
        for raw in ("", "x" * 181, "API key abc", "password rahsia",
                    "secret token", "recovery phrase", "pin bank 1234"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    validate_new_note(raw)
        with self.assertRaises(ValueError):
            serialize_notes(["x"] * (MAX_NOTES + 1))

    async def test_save_and_read_with_fake_telegram_bot(self):
        class FakeBot:
            def __init__(self):
                self.message = None
                self.calls = []
            async def get_chat(self, chat_id):
                return SimpleNamespace(pinned_message=self.message)
            async def send_message(self, chat_id, text):
                self.calls.append("send")
                self.message = SimpleNamespace(message_id=57, text=text)
                return self.message
            async def pin_chat_message(self, chat_id, message_id, disable_notification):
                self.calls.append("pin")
            async def edit_message_text(self, chat_id, message_id, text):
                self.calls.append("edit")
                self.message = SimpleNamespace(message_id=message_id, text=text)
        bot = FakeBot()
        initial, msg_id = await get_pinned_notes(bot, 1234)
        self.assertEqual((initial,msg_id), ([],None))
        await write_pinned_notes(bot, 1234, ["Catatan pertama"], None)
        self.assertEqual(bot.calls, ["send","pin"])
        notes, msg_id = await get_pinned_notes(bot, 1234)
        self.assertEqual(notes, ["Catatan pertama"])
        self.assertEqual(msg_id, 57)
        await write_pinned_notes(bot, 1234, [], msg_id)
        self.assertEqual((await get_pinned_notes(bot, 1234))[0], [])
        self.assertEqual(bot.calls[-1], "edit")

    async def test_unrelated_pinned_note_not_overwritten(self):
        class FakeBot:
            async def get_chat(self, chat_id):
                return SimpleNamespace(pinned_message=SimpleNamespace(
                    message_id=99, text="Unrelated pinned item"))
        notes, msg_id = await get_pinned_notes(FakeBot(), 123)
        self.assertIsNone(notes)
        self.assertIsNone(msg_id)


if __name__ == "__main__":
    unittest.main()
