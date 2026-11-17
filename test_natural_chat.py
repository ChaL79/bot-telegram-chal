"""Offline tests for natural, temporary, owner-only voice/text conversation."""
import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from natural_chat import (
    HISTORY_KEY, MAX_EXCHANGES, IDLE_TTL_SECONDS, NATURAL_CHAT_PROMPT,
    read_conversation, record_conversation, clear_conversation,
    build_chat_messages, valid_ai_response,
)


class NaturalContextTests(unittest.IsolatedAsyncioTestCase):
    def test_turns_are_shared_capped_and_expire(self):
        state = {}
        record_conversation(state, "Saya kerja offshore", "Boleh kita bincangkan.", now=100)
        previous = read_conversation(state, now=101)
        self.assertEqual(previous[0]["content"], "Saya kerja offshore")
        next_call = build_chat_messages(
            "Kenapa begitu?", [], previous, lambda notes: "")
        self.assertEqual([x["role"] for x in next_call],
                         ["system", "user", "assistant", "user"])
        self.assertIn("offshore", next_call[1]["content"])
        for i in range(20):
            record_conversation(state, "Soalan " + str(i), "Jawapan " + str(i), now=100+i)
        self.assertEqual(len(read_conversation(state, now=122)), 2*MAX_EXCHANGES)
        self.assertEqual(read_conversation(state, now=125+IDLE_TTL_SECONDS), [])
        self.assertNotIn(HISTORY_KEY, state)

    def test_failed_responses_and_bad_history_never_reused(self):
        state = {}
        record_conversation(state, "Hi", "Hello", now=10)
        self.assertFalse(valid_ai_response("ERROR", "ERROR"))
        self.assertFalse(valid_ai_response("Maaf, respons mengambil masa terlalu lama.", "ERROR"))
        self.assertTrue(valid_ai_response("Jawapan berguna", "ERROR"))
        state[HISTORY_KEY].append({"role": "system", "content": "malicious"})
        self.assertEqual(read_conversation(state, now=12), [])
        self.assertFalse(clear_conversation(state))

    async def test_actual_shared_reply_function_uses_history_and_serializes(self):
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
        selected = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                    and n.name in ("reply_with_context", "newchat_command")]
        self.assertEqual(len(selected), 2)
        received = []
        async def ask_ai(messages):
            received.append(messages)
            return "Jawapan " + str(len(received))
        ns = {
            "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "Update": object, "asyncio": asyncio,
            "read_user_notes": AsyncMock(return_value=["Suka muzik rock"]),
            "read_conversation": read_conversation,
            "record_conversation": record_conversation,
            "clear_conversation": clear_conversation,
            "build_chat_messages": build_chat_messages,
            "notes_as_instruction": lambda notes: " Nota: " + ", ".join(notes),
            "valid_ai_response": valid_ai_response,
            "FINAL_ERROR_MESSAGE": "ERROR",
            "log": logging.getLogger("natural-chat-test"),
            "ask_ai": ask_ai,
            "is_private_owner_update": lambda u: u.effective_user.id == 55,
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), "bot.py", "exec"), ns)
        state = {}
        context = SimpleNamespace(chat_data=state)
        first = await ns["reply_with_context"](context, 55, "Saya suka muzik rock.")
        second = await ns["reply_with_context"](context, 55, "Apa yang saya suka?")
        self.assertEqual(first, "Jawapan 1")
        self.assertEqual(second, "Jawapan 2")
        self.assertEqual(received[1][1]["content"], "Saya suka muzik rock.")
        self.assertEqual(received[1][2]["content"], "Jawapan 1")
        self.assertEqual(received[1][-1]["content"], "Apa yang saya suka?")
        self.assertIn("muzik rock", received[1][0]["content"])
        bad = SimpleNamespace(
            effective_user=SimpleNamespace(id=999),
            effective_message=SimpleNamespace(reply_text=AsyncMock()))
        await ns["newchat_command"](bad, context)
        self.assertEqual(len(read_conversation(state)), 4)
        good = SimpleNamespace(
            effective_user=SimpleNamespace(id=55),
            effective_message=SimpleNamespace(reply_text=AsyncMock()))
        await ns["newchat_command"](good, context)
        self.assertEqual(read_conversation(state), [])
        good.effective_message.reply_text.assert_awaited_once()

    async def test_errors_are_not_stored_and_no_private_content_logged(self):
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
        node = next(n for n in tree.body
                    if isinstance(n, ast.AsyncFunctionDef)
                    and n.name == "reply_with_context")
        ns = {
            "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "asyncio": asyncio,
            "read_user_notes": AsyncMock(return_value=[]),
            "read_conversation": read_conversation,
            "record_conversation": record_conversation,
            "build_chat_messages": build_chat_messages,
            "notes_as_instruction": lambda notes: "",
            "valid_ai_response": valid_ai_response,
            "FINAL_ERROR_MESSAGE": "ERROR",
            "log": logging.getLogger("natural-error-test"),
            "ask_ai": AsyncMock(return_value="ERROR"),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "bot.py", "exec"), ns)
        state = {}
        answer = await ns["reply_with_context"](SimpleNamespace(chat_data=state), 55, "Private content")
        self.assertEqual(answer, "ERROR")
        self.assertEqual(read_conversation(state), [])
        self.assertIn("soalan susulan", NATURAL_CHAT_PROMPT)


if __name__ == "__main__":
    unittest.main()
