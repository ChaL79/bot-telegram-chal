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


if __name__ == "__main__":
    unittest.main()
