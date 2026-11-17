"""Behavioral tests for new Telegram handlers without credentials or network."""
import ast
import asyncio
import logging
import os
from pathlib import Path
from pathlib import PurePath
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from memory_notes import get_pinned_notes, write_pinned_notes, validate_new_note, MAX_NOTES
from media_tools import MAX_IMAGE_BYTES, MAX_DOCUMENT_BYTES, ALLOWED_DOCUMENT_EXTENSIONS


def extract_handlers(*names, extras=None):
    tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in names]
    assert len(funcs) == len(names)
    namespace = {
        "Update": object,
        "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
        "asyncio": asyncio,
        "os": os,
        "PurePath": PurePath,
        "SYSTEM_PROMPT": "Test assistant",
        "MAX_NOTES": MAX_NOTES,
        "MAX_IMAGE_BYTES": MAX_IMAGE_BYTES,
        "MAX_DOCUMENT_BYTES": MAX_DOCUMENT_BYTES,
        "ALLOWED_DOCUMENT_EXTENSIONS": ALLOWED_DOCUMENT_EXTENSIONS,
        "get_pinned_notes": get_pinned_notes,
        "write_pinned_notes": write_pinned_notes,
        "validate_new_note": validate_new_note,
        "log": logging.getLogger("handler-test"),
        "is_private_owner_update": lambda update: update.effective_user.id == 321,
    }
    namespace.update(extras or {})
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "bot.py", "exec"),
         namespace)
    return namespace


class FakeMemoryBot:
    def __init__(self):
        self.pinned = None
        self.pin_count = 0
    async def get_chat(self, chat_id):
        return SimpleNamespace(pinned_message=self.pinned)
    async def send_message(self, chat_id, text):
        self.pinned = SimpleNamespace(message_id=111, text=text)
        return self.pinned
    async def pin_chat_message(self, chat_id, message_id, disable_notification=True):
        self.pin_count += 1
    async def edit_message_text(self, chat_id, message_id, text):
        self.pinned = SimpleNamespace(message_id=message_id, text=text)


class MemoryHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_can_save_list_delete_and_clear_pinned_notes(self):
        ns = extract_handlers("personal_memory_command")
        bot = FakeMemoryBot()
        state = {}
        async def issue(user, command, *args):
            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=user),
                effective_chat=SimpleNamespace(id=321, type="private"),
                effective_message=SimpleNamespace(
                    text="/" + command,
                    reply_text=AsyncMock()))
            ctx = SimpleNamespace(bot=bot, chat_data=state, args=list(args))
            await ns["personal_memory_command"](update, ctx)
            return update.effective_message.reply_text

        reject = await issue(999, "remember", "Not allowed")
        reject.assert_not_awaited()
        self.assertIsNone(bot.pinned)
        accepted = await issue(321, "remember", "Saya suka rock music")
        self.assertIn("disimpan", accepted.await_args.args[0])
        self.assertEqual(bot.pin_count, 1)
        self.assertEqual((await get_pinned_notes(bot, 321))[0], ["Saya suka rock music"])
        view = await issue(321, "memories")
        self.assertIn("rock music", view.await_args.args[0])
        bad = await issue(321, "remember", "my API key is abc")
        self.assertIn("Jangan simpan", bad.await_args.args[0])
        deleted = await issue(321, "forget", "1")
        self.assertIn("dipadam", deleted.await_args.args[0])
        self.assertEqual((await get_pinned_notes(bot, 321))[0], [])
        await issue(321, "remember", "Tolong ringkaskan jawapan")
        await issue(321, "clearmemory")
        self.assertEqual((await get_pinned_notes(bot, 321))[0], [])

    async def test_does_not_overwrite_other_pinned_messages(self):
        ns = extract_handlers("personal_memory_command")
        bot = FakeMemoryBot()
        bot.pinned = SimpleNamespace(message_id=88, text="Important unrelated pinned")
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=321),
            effective_chat=SimpleNamespace(id=321, type="private"),
            effective_message=SimpleNamespace(text="/remember", reply_text=AsyncMock()))
        ctx = SimpleNamespace(bot=bot, chat_data={}, args=["Personal", "note"])
        await ns["personal_memory_command"](update, ctx)
        self.assertEqual(bot.pinned.text, "Important unrelated pinned")
        self.assertIn("pinned lain", update.effective_message.reply_text.await_args.args[0])


class MediaHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_photo_checks_owner_and_invokes_vision(self):
        image = b"\x89PNG\r\n\x1a\nDATA"
        vision = AsyncMock(return_value="Saya nampak carta.")
        file = SimpleNamespace(file_size=len(image), download_as_bytearray=AsyncMock(return_value=bytearray(image)))
        bot = SimpleNamespace(send_chat_action=AsyncMock(), get_file=AsyncMock(return_value=file))
        ns = extract_handlers("handle_photo", extras={"analyze_image":vision,
            "GROQ_API_KEY":"key", "log":logging.getLogger("photo-test")})
        def update(user):
            return SimpleNamespace(
                effective_user=SimpleNamespace(id=user),
                effective_chat=SimpleNamespace(id=321, type="private"),
                effective_message=SimpleNamespace(
                    photo=[SimpleNamespace(file_id="abc", file_size=len(image))],
                    caption="Apa ini?", reply_text=AsyncMock()))
        ctx = SimpleNamespace(bot=bot)
        bad = update(999)
        await ns["handle_photo"](bad, ctx)
        bot.get_file.assert_not_awaited()
        good = update(321)
        await ns["handle_photo"](good, ctx)
        vision.assert_awaited_once()
        good.effective_message.reply_text.assert_awaited_once_with("Saya nampak carta.")

    async def test_documents_guard_and_txt_extraction_route(self):
        doc_bytes = b"Example document"
        source = SimpleNamespace(file_size=len(doc_bytes),
                                 download_as_bytearray=AsyncMock(return_value=bytearray(doc_bytes)))
        bot = SimpleNamespace(send_chat_action=AsyncMock(),get_file=AsyncMock(return_value=source))
        answer = AsyncMock(return_value="Ringkasan: contoh dokumen.")
        ns = extract_handlers("handle_document",extras={
            "ask_ai":answer, "GROQ_API_KEY":"key",
            "MODEL_DEFAULT":"gemini-3.5-flash-lite",
            "extract_document_text":lambda content,ext: content.decode(),
            "analyze_image":AsyncMock(), "analyze_pdf":AsyncMock(),
            "log":logging.getLogger("doc-test"),
        })
        def update(user,filename):
            return SimpleNamespace(
                effective_user=SimpleNamespace(id=user),
                effective_chat=SimpleNamespace(id=321, type="private"),
                effective_message=SimpleNamespace(
                    document=SimpleNamespace(file_name=filename,
                        file_size=len(doc_bytes),file_id="file"),
                    caption=None, reply_text=AsyncMock()))
        ctx = SimpleNamespace(bot=bot)
        await ns["handle_document"](update(999,"sample.txt"),ctx)
        bot.get_file.assert_not_awaited()
        allowed = update(321,"notes.txt")
        await ns["handle_document"](allowed,ctx)
        answer.assert_awaited_once()
        allowed.effective_message.reply_text.assert_awaited_once_with(
            "Ringkasan: contoh dokumen.")


class MemoryCommandGuidanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_instruction_prefix_is_not_saved_as_note_and_shows_exact_syntax(self):
        import re
        ns = extract_handlers("handle_message", extras={
            "re": re,
            "ask_ai": AsyncMock(return_value="regular model reply"),
            "read_user_notes": AsyncMock(return_value=[]),
            "notes_as_instruction": lambda notes: "",
            "SYSTEM_PROMPT": "You are helpful",
            "FINAL_ERROR_MESSAGE": "fail",
        })
        msg = SimpleNamespace(
            text="Hantar /remember Saya suka muzik rock klasik",
            reply_text=AsyncMock())
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=321),
            effective_chat=SimpleNamespace(id=321, type="private"),
            effective_message=msg)
        ctx = SimpleNamespace(
            bot=SimpleNamespace(send_chat_action=AsyncMock()))
        await ns["handle_message"](update, ctx)
        msg.reply_text.assert_awaited_once()
        self.assertIn("mesti BERMULA dengan /remember", msg.reply_text.await_args.args[0])
        ns["ask_ai"].assert_not_awaited()
        ctx.bot.send_chat_action.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
