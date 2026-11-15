import asyncio
import json
import unittest
import httpx
from gemini_advanced import (
    POWER_MODEL, RESEARCH_MODEL, MODE_LIMITS, quota_reserve,
    safe_web_sources, split_telegram_text, run_advanced,
)


class AdvancedTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepdive_uses_grounding_and_real_sources(self):
        requests = []
        def handle(req):
            requests.append(json.loads(req.content))
            return httpx.Response(200, json={"candidates": [{
                "content": {"parts": [{"text": "Sumber rasmi telah disemak."}]},
                "groundingMetadata": {"groundingChunks": [
                    {"web": {"title": "Official", "uri": "https://example.org/report"}},
                    {"web": {"title": "Official", "uri": "https://example.org/report"}},
                ]},
            }]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as cli:
            answer, ok = await run_advanced("Periksa laporan", "TEST_KEY", "deepdive", client=cli)
        self.assertTrue(ok)
        self.assertIn("Sumber daripada Google Search", answer)
        self.assertEqual(answer.count("https://example.org/report"), 1)
        self.assertEqual(requests[0]["tools"], [{"google_search": {}}])
        self.assertEqual(requests[0]["generationConfig"]["thinkingConfig"], {"thinkingBudget": 512})

    async def test_power_uses_high_thinking_no_web(self):
        seen=[]
        def handler(req):
            seen.append((str(req.url), json.loads(req.content)))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text":"Analisis selesai"}]}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
            reply, ok = await run_advanced("Audit Python", "TEST_KEY", "power", client=cli)
        self.assertTrue(ok)
        self.assertIn(POWER_MODEL, seen[0][0])
        self.assertEqual(seen[0][1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "high"})
        self.assertNotIn("tools", seen[0][1])

    async def test_429_does_not_retry(self):
        count=0
        def response(req):
            nonlocal count
            count+=1
            return httpx.Response(429)
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as cli:
            answer, ok=await run_advanced("Harga", "TEST_KEY", "deepdive", client=cli)
        self.assertFalse(ok)
        self.assertEqual(count,1)
        self.assertIn("Free Tier",answer)

    async def test_research_no_evidence_warning(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200,json={"candidates":[{"content":{"parts":[{"text":"Saya rasa begini"}]}}]})
        )) as cli:
            answer, ok = await run_advanced("isu baru", "TEST_KEY", "deepdive", client=cli)
        self.assertTrue(ok)
        self.assertIn("Tiada pautan sumber",answer)

    async def test_unsupported_model_fails_closed(self):
        for status in (400,404,403):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(status)
            )) as cli:
                answer, ok = await run_advanced("Penyelidikan", "TEST_KEY", "deepdive", client=cli)
            self.assertFalse(ok)
            self.assertNotIn("Sumber daripada Google Search",answer)

    def test_sources_reject_invalid_uris(self):
        data={"candidates":[{"groundingMetadata":{"groundingChunks":[
            {"web":{"uri":"javascript:alert(1)","title":"evil"}},
            {"web":{"uri":"https://example.net/ok","title":"Site"}},
            {"web":{"uri":"https://user:pw@example.net/secret","title":"bad"}},
        ]}}]}
        self.assertEqual(safe_web_sources(data),[("Site","https://example.net/ok")])

    def test_quota_rpm_daily_and_isolation(self):
        a,b={},{}
        self.assertEqual(MODE_LIMITS["power"],(2,12))
        self.assertTrue(quota_reserve(a,"power",monotonic_now=1,utc_day="2026-11-15")[0])
        self.assertTrue(quota_reserve(a,"power",monotonic_now=2,utc_day="2026-11-15")[0])
        self.assertFalse(quota_reserve(a,"power",monotonic_now=3,utc_day="2026-11-15")[0])
        self.assertTrue(quota_reserve(b,"power",monotonic_now=3,utc_day="2026-11-15")[0])
        for i in range(10):
            self.assertTrue(quota_reserve(a,"power",monotonic_now=65+i*61,utc_day="2026-11-15")[0])
        self.assertFalse(quota_reserve(a,"power",monotonic_now=999,utc_day="2026-11-15")[0])
        self.assertTrue(quota_reserve(a,"power",monotonic_now=1000,utc_day="2026-11-16")[0])

    def test_telegram_split_below_limit(self):
        parts=split_telegram_text("x"*7500)
        self.assertGreater(len(parts),1)
        self.assertTrue(all(len(p)<=3800 for p in parts))
        self.assertEqual("".join(parts),"x"*7500)

    async def test_memory_history_sent_to_research(self):
        seen=[]
        def mock(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200,json={"candidates":[{"content":{"parts":[{"text":"Baik"}]}}]})
        history=[
            {"role":"user","parts":[{"text":"MELUR"}]},
            {"role":"model","parts":[{"text":"Ingat MELUR"}]},
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            answer, ok=await run_advanced("Apa tadi?","TEST_KEY","deepdive",history=history,client=cli)
        self.assertTrue(ok)
        self.assertEqual([x["role"] for x in seen[0]["contents"]],["user","model","user"])

    async def test_invalid_history_rejected_before_api(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: self.fail("Should not reach API")
        )) as cli:
            with self.assertRaises(ValueError):
                await run_advanced("Hello","TEST_KEY","power",history=[{"role":"model","parts":[{"text":"x"}]}],client=cli)


if __name__ == "__main__":
    unittest.main()
