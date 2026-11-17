"""Cleanup of private temporary audio from Telegram voice notes."""
import logging
import os

log = logging.getLogger("voice_files")


def cleanup_voice_tempfiles(*paths: str) -> None:
    """Best-effort cleanup without printing private filenames or transcripts."""
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("Temporary voice file cleanup failed")
