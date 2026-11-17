"""Fail-closed owner checks on both Groq/OpenRouter-backed entrypoints.

Loads the actual AST handler definitions; Telegram credentials and paid API keys
are never required, and no external network calls are performed.
"""
import ast
import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from gemini_extension import is_authorized_owner


def handlers_under_test():
    tree = ast.parse(Path(__file__).with_name("bot.py").read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("is_private_owner_update", "start", "myid", "handle_message", "handle_voice")
    ]
    assert len(selected) == 5, "All protected production handlers must be present"
    namespace = {
        "Update": object,
        "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
        "is_authorized_owner": is_authorized_owner,
        "os": os, "asyncio": asyncio,
        "SYSTEM_PROMPT": "Test only",
        "FINAL_ERROR_MESSAGE": "Service unavailable",
        "log": logging.getLogger("owner_access_test"),
        "ask_ai": AsyncMock(return_value="Authorized answer"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 filename="bot.py", mode="exec"), namespace)
    return namespace


def fake_update(user_id, chat_type="private"):
    msg = SimpleNamespace(
        text="hello", voice=SimpleNamespace(file_id="forbidden-file"),
        reply_text=AsyncMock())
    user = SimpleNamespace(id=user_id) if user_id is not None else None
    chat = SimpleNamespace(id=42, type=chat_type)
    update = SimpleNamespace(effective_message=msg, effective_user=user,
                             effective_chat=chat)
    client = SimpleNamespace(
        send_chat_action=AsyncMock(),
        get_file=AsyncMock(side_effect=AssertionError("media fetched for unauthorized user")))
    ctx = SimpleNamespace(bot=client)
    return update, ctx


class OwnerOnlyAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_gate_rejects_unset_id_wrong_user_and_groups(self):
        ns = handlers_under_test()
        for owner, user, chat_type in [
            ("", 1234, "private"),
            ("abc", 1234, "private"),
            ("1234", 9999, "private"),
            ("1234", 1234, "group"),
            ("1234", 1234, "supergroup"),
            ("1234", None, "private"),
        ]:
            with self.subTest(owner=owner, user=user, chat_type=chat_type):
                update, _ = fake_update(user, chat_type)
                with patch.dict(os.environ, {"GEMINI_OWNER_ID": owner}):
                    self.assertFalse(ns["is_private_owner_update"](update))

    async def test_text_and_voice_reject_unauthorized_before_any_side_effect(self):
        ns = handlers_under_test()
        for handler in ("handle_message", "handle_voice"):
            for owner, user, chat_type in [
                ("1234", 9999, "private"),
                ("1234", 1234, "group"),
                ("", 1234, "private"),
            ]:
                with self.subTest(handler=handler, owner=owner, user=user, chat=chat_type):
                    update, ctx = fake_update(user, chat_type)
                    with patch.dict(os.environ, {"GEMINI_OWNER_ID": owner}):
                        await ns[handler](update, ctx)
                    ctx.bot.send_chat_action.assert_not_awaited()
                    ctx.bot.get_file.assert_not_awaited()
                    ns["ask_ai"].assert_not_awaited()
                    update.effective_message.reply_text.assert_not_awaited()
                    ns["ask_ai"].reset_mock()

    async def test_public_commands_are_owner_only(self):
        ns = handlers_under_test()
        for handler in ("start", "myid"):
            for owner, user, chat_type in [
                ("1234", 9999, "private"),
                ("1234", 1234, "group"),
                ("1234", 1234, "supergroup"),
                ("", 1234, "private"),
                ("invalid", 1234, "private"),
                ("1234", None, "private"),
            ]:
                with self.subTest(handler=handler, owner=owner, user=user, chat=chat_type):
                    update, ctx = fake_update(user, chat_type)
                    with patch.dict(os.environ, {"GEMINI_OWNER_ID": owner}):
                        await ns[handler](update, ctx)
                    update.effective_message.reply_text.assert_not_awaited()

    async def test_owner_can_still_use_start_and_myid(self):
        ns = handlers_under_test()
        for handler in ("start", "myid"):
            update, ctx = fake_update(1234)
            with patch.dict(os.environ, {"GEMINI_OWNER_ID": "1234"}):
                await ns[handler](update, ctx)
            update.effective_message.reply_text.assert_awaited_once()
            self.assertTrue(update.effective_message.reply_text.await_args.args[0])
            if handler == "myid":
                self.assertIn("1234", update.effective_message.reply_text.await_args.args[0])

    async def test_legitimate_owner_text_still_works(self):
        ns = handlers_under_test()
        update, ctx = fake_update(1234)
        with patch.dict(os.environ, {"GEMINI_OWNER_ID": "1234"}):
            self.assertTrue(ns["is_private_owner_update"](update))
            await ns["handle_message"](update, ctx)
        ctx.bot.send_chat_action.assert_awaited_once()
        ns["ask_ai"].assert_awaited_once()
        update.effective_message.reply_text.assert_awaited_once_with("Authorized answer")


if __name__ == "__main__":
    unittest.main()
