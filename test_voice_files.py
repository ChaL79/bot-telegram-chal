"""Regression checks for privacy-sensitive Telegram voice-note handling."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from voice_files import cleanup_voice_tempfiles


class VoicePrivacyTests(unittest.TestCase):
    def test_cleanup_success_and_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / "recording.oga", Path(folder) / "recording.wav"]
            for p in paths:
                p.write_bytes(b"private audio")
            cleanup_voice_tempfiles(*(str(p) for p in paths))
            self.assertTrue(all(not p.exists() for p in paths))
            cleanup_voice_tempfiles(*(str(p) for p in paths))

    def test_cleanup_permission_failure_does_not_log_private_filename(self):
        private_path = "/tmp/private_voice_file_id.wav"
        with patch("voice_files.os.unlink", side_effect=PermissionError("private")):
            with self.assertLogs("voice_files", level="WARNING") as output:
                cleanup_voice_tempfiles(private_path)
        self.assertNotIn(private_path, "\n".join(output.output))

    def test_handler_cleanup_in_finally_and_no_transcript_logging(self):
        code = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
        parsed = ast.parse(code)
        handler = next(x for x in parsed.body if isinstance(x, ast.AsyncFunctionDef)
                       and x.name == "handle_voice")
        self.assertTrue(any(
            isinstance(n, ast.Try) and any(
                isinstance(q, ast.Call) and isinstance(q.func, ast.Name)
                and q.func.id == "cleanup_voice_tempfiles"
                for statement in n.finalbody for q in ast.walk(statement)
            ) for n in ast.walk(handler)))
        for n in ast.walk(handler):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr in ("info", "warning", "error", "debug", "exception"):
                    self.assertFalse(any(isinstance(a, ast.Name) and a.id == "transcript"
                                         for a in n.args))



class VoiceHandlerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Execute the ACTUAL AST function with fake Telegram/audio dependencies.

    The CI environment deliberately does not import bot.py, which requires
    the Telegram package and runtime credentials. The extracted handler is
    executed with fully mocked dependencies and real temporary cleanup.
    """
    async def _run_handler(self, *, transcription="hello", download_error=False,
                           reply_error=False):
        import asyncio
        import logging
        import uuid
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        src = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "handle_voice")
        paths = []
        async def download(dest):
            paths.append(dest)
            if download_error:
                raise OSError("test download blocked")
            Path(dest).write_bytes(b"short fake input")
        def export(dest, format):
            self.assertEqual(format, "wav")
            paths.append(dest)
            Path(dest).write_bytes(b"short fake converted audio")
        class FakeAudio:
            def export(self, dest, format):
                export_result = export(dest, format)
                return export_result
        fake_source = SimpleNamespace(download_to_drive=download)
        message = SimpleNamespace(
            voice=SimpleNamespace(file_id="voice-test-id",
                                  file_unique_id="private-file-unique-id"),
            reply_text=AsyncMock(side_effect=RuntimeError("failed reply")
                                 if reply_error else None))
        bot = SimpleNamespace(
            send_chat_action=AsyncMock(),
            get_file=AsyncMock(return_value=fake_source))
        ctx = SimpleNamespace(bot=bot)
        upd = SimpleNamespace(effective_message=message,
                              effective_chat=SimpleNamespace(id=12345))
        simulated_ask_ai = AsyncMock(return_value="test answer")
        simulated_transcribe = AsyncMock(return_value=transcription)
        namespace = {
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "asyncio": asyncio, "uuid": uuid,
            "AudioSegment": SimpleNamespace(from_file=lambda *args, **kw: FakeAudio()),
            "SYSTEM_PROMPT": "test", "FINAL_ERROR_MESSAGE": "fallback",
            "transcribe_voice": simulated_transcribe, "ask_ai": simulated_ask_ai,
            "read_user_notes": AsyncMock(return_value=[]),
            "notes_as_instruction": lambda notes: "",
            "send_spoken_reply": AsyncMock(return_value=True),
            "reply_with_context": simulated_ask_ai,
            "valid_ai_response": lambda response, error: response != error,
            "cleanup_voice_tempfiles": cleanup_voice_tempfiles,
            "is_private_owner_update": lambda update: True,
            "log": logging.getLogger("voice_handler_test"),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]),
                     filename="bot.py", mode="exec"), namespace)
        try:
            if reply_error and not download_error and transcription:
                with self.assertRaises(RuntimeError):
                    await namespace["handle_voice"](upd, ctx)
            else:
                await namespace["handle_voice"](upd, ctx)
        finally:
            self.assertTrue(all(not Path(p).exists() for p in paths),
                            "Audio artifacts must be removed on every exit")
        return message, simulated_transcribe, simulated_ask_ai, paths

    async def test_successful_voice_message_cleans_both_audio_files(self):
        message, transcribe, ask_ai, paths = await self._run_handler()
        self.assertEqual(len(paths), 2)
        self.assertTrue(all("/chal_voice_" in p for p in paths))
        self.assertTrue(all("private-file-unique-id" not in p for p in paths))
        transcribe.assert_awaited_once()
        ask_ai.assert_awaited_once()
        message.reply_text.assert_awaited_once_with("test answer")

    async def test_failed_download_returns_safely_without_leaking_audio(self):
        message, transcribe, ask_ai, paths = await self._run_handler(
            download_error=True)
        self.assertEqual(len(paths), 1)
        transcribe.assert_not_awaited()
        ask_ai.assert_not_awaited()
        message.reply_text.assert_awaited_once_with(
            "Maaf, saya tidak dapat memproses fail audio ini.")

    async def test_missing_transcription_cleans_audio(self):
        message, transcribe, ask_ai, paths = await self._run_handler(
            transcription=None)
        self.assertEqual(len(paths), 2)
        transcribe.assert_awaited_once()
        ask_ai.assert_not_awaited()
        message.reply_text.assert_awaited_once_with(
            "Maaf, saya tidak dapat memahami voice note ini. Cuba lagi.")

    async def test_failed_telegram_reply_still_erases_audio(self):
        message, transcribe, ask_ai, paths = await self._run_handler(
            reply_error=True)
        self.assertEqual(len(paths), 2)
        transcribe.assert_awaited_once()
        ask_ai.assert_awaited_once()

if __name__ == "__main__":
    unittest.main()
