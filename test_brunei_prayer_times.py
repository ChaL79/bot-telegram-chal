"""Offline, date-verified Brunei prayer time lookup regression tests."""
import ast
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from brunei_prayer_times import (
    TIMEZONE, recognize_prayer_question, requested_date, requested_zones,
    prayer_requested, lookup_brunei_prayer_time, _valid_schedule,
)

DAY = {
    "date": "2026-11-18", "imsak": "04:37", "fajr": "04:47",
    "syuruk": "06:09", "dhuha": "06:32", "dhuhr": "12:06",
    "asr": "15:28", "maghrib": "18:02", "isha": "19:15",
}
NOW = dt.datetime(2026, 11, 18, 5, 16, tzinfo=TIMEZONE)


class BruneiPrayerTimeTests(unittest.IsolatedAsyncioTestCase):
    def test_intent_and_zones_are_restricted(self):
        self.assertTrue(recognize_prayer_question("waktu subuh brunei hari ini? 5.55"))
        self.assertTrue(recognize_prayer_question("jadual waktu solat Belait esok"))
        self.assertFalse(recognize_prayer_question("deep research trading gold M5"))
        self.assertFalse(recognize_prayer_question("waktu subuh Tokyo hari ini"))
        self.assertFalse(recognize_prayer_question("Brunei tempat bercuti di mana?"))
        self.assertEqual(requested_zones("Subuh di Seria"), ["BRN03"])
        self.assertEqual(requested_zones("Subuh Belait dan Tutong"), ["BRN02", "BRN03"])
        self.assertEqual(len(requested_zones("Subuh Brunei")), 4)

    def test_local_date_and_bad_dates(self):
        self.assertEqual(requested_date("subuh hari ini", now=NOW), dt.date(2026,11,18))
        self.assertEqual(requested_date("subuh esok", now=NOW), dt.date(2026,11,19))
        self.assertEqual(requested_date("subuh 18/11/2026", now=NOW), dt.date(2026,11,18))
        self.assertEqual(requested_date("subuh 2026-11-18", now=NOW), dt.date(2026,11,18))
        self.assertEqual(requested_date("subuh 18 November 2026", now=NOW), dt.date(2026,11,18))
        self.assertIsNone(requested_date("Subuh 31/02/2026", now=NOW))
        self.assertEqual(prayer_requested("Waktu subuh Brunei"), ("fajr", "Subuh"))

    def test_reject_wrong_date_missing_fields_and_unvalidated_time(self):
        date = dt.date(2026, 11, 18)
        self.assertEqual(_valid_schedule([DAY], date), DAY)
        self.assertIsNone(_valid_schedule([DAY], dt.date(2026, 11, 17)))
        self.assertIsNone(_valid_schedule([dict(DAY, fajr="99:00")], date))
        self.assertIsNone(_valid_schedule([dict(DAY, dhuhr=None)], date))
        self.assertIsNone(_valid_schedule([DAY, DAY], date))

    async def test_user_screenshot_query_uses_dated_source_not_news(self):
        requested = []
        def mock(req):
            url = str(req.url)
            requested.append(url)
            self.assertTrue(
                req.url.host in {"simplesolat-data.netlify.app",
                                 "raw.githubusercontent.com"})
            self.assertIn("/prayer-times/BN/", url)
            self.assertIn("/2026-11.json", url)
            return httpx.Response(200, json=[DAY])
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            text = await lookup_brunei_prayer_time(
                "waktu subuh brunei hari ini? 5.55", client=client, now=NOW)
        self.assertIn("18/11/2026", text)
        self.assertIn("Brunei-Muara: 04:47 pagi", text)
        self.assertIn("Sumber data: cermin komuniti", text)
        self.assertIn("bukan capaian langsung", text)
        self.assertEqual(len(requested), 4)

    async def test_backup_and_no_guesses_for_unavailable_or_wrong_day(self):
        seen = []
        def response(req):
            seen.append(str(req.url))
            if req.url.host == "simplesolat-data.netlify.app":
                return httpx.Response(503)
            return httpx.Response(200, json=[DAY])
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            answer = await lookup_brunei_prayer_time(
                "subuh belait 18 Nov 2026", client=client, now=NOW)
        self.assertIn("Belait: 04:47 pagi", answer)
        self.assertEqual(len(seen), 2)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(
                200, json=[dict(DAY, date="2026-11-17")]))
        ) as client:
            answer = await lookup_brunei_prayer_time(
                "subuh Brunei-Muara hari ini", client=client, now=NOW)
        self.assertIn("tidak dapat disahkan", answer)
        self.assertNotIn("04:47 pagi", answer)

    async def test_owner_only_handler_stays_guarded_before_source_requests(self):
        source = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        handler = next(
            n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "advanced_command"
        )
        code = ast.unparse(handler)
        self.assertLess(code.index("is_authorized_owner"),
                        code.index("lookup_brunei_prayer_time"))
        self.assertLess(code.index("lookup_brunei_prayer_time"),
                        code.index("quota_reserve"))
        state = {"called": 0}
        async def lookup(q):
            state["called"] += 1
            return "Brunei-Muara: 04:47 pagi."
        ns = {
            "Update": object, "ContextTypes": SimpleNamespace(DEFAULT_TYPE=object),
            "os": __import__("os"), "asyncio": __import__("asyncio"),
            "MAX_INPUT_CHARS": 4000,
            "is_authorized_owner": lambda uid, owner, chat_type:
                 uid == 77 and chat_type == "private",
            "recognize_prayer_question": recognize_prayer_question,
            "lookup_brunei_prayer_time": lookup,
            "split_telegram_text": lambda text: [text],
            "quota_reserve": lambda state, mode:
                 (_ for _ in ()).throw(AssertionError("quota must not be charged")),
            "log": __import__("logging").getLogger("brunei-test"),
        }
        exec(compile(ast.Module(body=[handler], type_ignores=[]),
                     "bot.py", "exec"), ns)
        def user(uid):
            message = SimpleNamespace(
                text="/deepdive waktu Subuh Brunei hari ini",
                reply_text=AsyncMock())
            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=uid),
                effective_chat=SimpleNamespace(id=77, type="private"),
                effective_message=message)
            context = SimpleNamespace(
                args=["waktu", "Subuh", "Brunei", "hari", "ini"],
                bot=SimpleNamespace(send_chat_action=AsyncMock()),
                chat_data={})
            return update, context
        denied, context = user(999)
        await ns["advanced_command"](denied, context)
        self.assertEqual(state["called"], 0)
        denied.effective_message.reply_text.assert_not_awaited()
        allowed, context = user(77)
        await ns["advanced_command"](allowed, context)
        self.assertEqual(state["called"], 1)
        allowed.effective_message.reply_text.assert_awaited_once()
        self.assertIn("04:47", allowed.effective_message.reply_text.await_args.args[0])


if __name__ == "__main__":
    unittest.main()
