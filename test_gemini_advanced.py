import json
import unittest
import httpx

from gemini_advanced import (
    POWER_MODEL, RESEARCH_MODEL, MODE_LIMITS, quota_reserve,
    split_telegram_text, run_advanced,
)


class AdvancedTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepdive_uses_public_metadata_and_real_links(self):
        seen = []
        def search_mock(req):
            if req.url.host == "www.bing.com":
                xml = b"""<rss><channel><item><title>Official Python</title>
                  <link>https://docs.python.org/3/</link>
                  <description>Python reference</description></item></channel></rss>"""
                return httpx.Response(200, content=xml)
            return httpx.Response(404)
        def ai_mock(req):
            seen.append((str(req.url), json.loads(req.content)))
            return httpx.Response(200, json={"candidates":[{"content":{"parts":[{"text":"Dokumentasi membincangkan Python."}]}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(search_mock)) as search_client:
            async with httpx.AsyncClient(transport=httpx.MockTransport(ai_mock)) as client:
                answer, ok = await run_advanced("Python reference", "TEST_KEY", "deepdive",
                                                client=client, search_client=search_client)
        self.assertTrue(ok)
        self.assertIn("https://docs.python.org/3/", answer)
        self.assertIn("Tidak semestinya membaca artikel penuh", answer)
        self.assertIn("docs.python.org", seen[0][1]["contents"][-1]["parts"][0]["text"])
        self.assertNotIn("tools", seen[0][1])
        self.assertIn(RESEARCH_MODEL, seen[0][0])

    async def test_deepdive_single_request_55s_timeout_no_retry(self):
        from gemini_advanced import GEMINI_READ_TIMEOUT
        calls = []
        def search_mock(req):
            if req.url.host == "www.bing.com":
                return httpx.Response(200, content=b"""<rss><channel><item>
                <title>Python asyncio technical reference</title>
                <link>https://docs.python.org/3/library/asyncio-sync.html</link>
                <description>Python asyncio semaphore documentation</description>
                </item></channel></rss>""")
            return httpx.Response(403)
        def api_mock(req):
            calls.append(str(req.url))
            raise httpx.ReadTimeout("read timed out", request=req)
        async with httpx.AsyncClient(transport=httpx.MockTransport(search_mock)) as search:
            async with httpx.AsyncClient(transport=httpx.MockTransport(api_mock)) as api:
                answer, ok = await run_advanced(
                    "Python asyncio semaphore", "TEST_KEY", "deepdive",
                    search_client=search, client=api)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertIn("masa tamat", answer)
        self.assertIn("BUKAN analisis Gemini", answer)
        self.assertEqual(GEMINI_READ_TIMEOUT["deepdive"], 55.0)
        self.assertEqual(GEMINI_READ_TIMEOUT["power"], 35.0)

    async def test_power_single_network_failure_no_retry(self):
        calls = []
        def api_mock(req):
            calls.append(req)
            raise httpx.ConnectError("network unavailable", request=req)
        async with httpx.AsyncClient(transport=httpx.MockTransport(api_mock)) as api:
            answer, ok = await run_advanced("Explain", "TEST_KEY", "power", client=api)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertIn("tidak dapat dihubungi", answer)
        self.assertNotIn("Petikan halaman", answer)

    async def test_503_preserves_sources_without_fake_ai_analysis_or_retry(self):
        requests = []
        def search_mock(req):
            if req.url.host == "www.bing.com":
                return httpx.Response(200, content=b"""<rss><channel><item>
                <title>Python asyncio technical documentation</title>
                <link>https://docs.python.org/3/library/asyncio-sync.html</link>
                <description>Python asyncio synchronization</description>
                </item></channel></rss>""")
            if req.url.host == "docs.python.org":
                return httpx.Response(200, headers={"Content-Type": "text/html"},
                    content=b"""<html><main><h1>Python asyncio Semaphore</h1>
                    <p>The Python asyncio Semaphore manages concurrent task access with a counter.</p>
                    </main></html>""")
            return httpx.Response(404)
        def api_mock(req):
            requests.append(req)
            return httpx.Response(503, json={"error": {"message": "Upstream overloaded"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(search_mock)) as search:
            async with httpx.AsyncClient(transport=httpx.MockTransport(api_mock)) as api:
                answer, ok = await run_advanced(
                    "Python asyncio Semaphore", "TEST_KEY", "deepdive",
                    search_client=search, client=api)
        self.assertFalse(ok)
        self.assertEqual(len(requests), 1)
        self.assertIn("HTTP 503", answer)
        self.assertIn("BUKAN analisis Gemini", answer)
        self.assertIn("docs.python.org", answer)
        self.assertIn("Petikan halaman diambil", answer)
        self.assertNotIn("Upstream overloaded", answer)

    async def test_power_http_503_has_no_fake_sources(self):
        calls = []
        def mock(req):
            calls.append(req)
            return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as api:
            answer, ok = await run_advanced("Hi", "TEST_KEY", "power", client=api)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("Petikan halaman", answer)

    async def test_power_uses_proven_flash_lite_no_thinking_high(self):
        seen = []
        def mock(req):
            seen.append((str(req.url), json.loads(req.content)))
            return httpx.Response(200,json={"candidates":[{"content":{"parts":[{"text":"Selesai"}]}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            answer, ok = await run_advanced("Audit asas", "TEST_KEY", "power", client=cli)
        self.assertTrue(ok)
        self.assertEqual(answer, "Selesai")
        self.assertIn("gemini-3.5-flash-lite", seen[0][0])
        self.assertEqual(POWER_MODEL, "gemini-3.5-flash-lite")
        self.assertNotIn("thinkingConfig", seen[0][1]["generationConfig"])
        self.assertNotIn("tools", seen[0][1])

    async def test_public_search_unavailable_no_fabricated_research(self):
        def search_mock(req):
            return httpx.Response(403)
        def forbidden_ai(req):
            self.fail("Should not call Gemini if no sources")
        async with httpx.AsyncClient(transport=httpx.MockTransport(search_mock)) as search_client:
            async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden_ai)) as client:
                answer, ok = await run_advanced("Berita terkini", "TEST_KEY", "deepdive",
                                                client=client, search_client=search_client)
        self.assertFalse(ok)
        self.assertIn("tidak boleh mengesahkan", answer)

    async def test_429_no_retry(self):
        count = 0
        def mock(req):
            nonlocal count
            count += 1
            return httpx.Response(429)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            answer, ok = await run_advanced("Audit", "TEST_KEY", "power", client=client)
        self.assertFalse(ok)
        self.assertEqual(count, 1)
        self.assertIn("Kuota", answer)

    async def test_invalid_history_blocked(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: self.fail("No API call allowed")
        )) as cli:
            with self.assertRaises(ValueError):
                await run_advanced("Hi", "TEST_KEY", "power",
                    history=[{"role":"model","parts":[{"text":"bad"}]}], client=cli)

    async def test_multi_turn_history_preserved(self):
        seen = []
        def mock(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200,json={"candidates":[{"content":{"parts":[{"text":"MELUR"}]}}]})
        history = [
            {"role":"user","parts":[{"text":"Ingat MELUR"}]},
            {"role":"model","parts":[{"text":"Baik MELUR"}]},
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            answer, ok = await run_advanced("Apa tadi?", "TEST_KEY", "power",
                                             history=history, client=cli)
        self.assertTrue(ok)
        self.assertEqual(answer,"MELUR")
        self.assertEqual([i["role"] for i in seen[0]["contents"]],["user","model","user"])

    def test_quota_rpm_daily(self):
        a = {}
        self.assertEqual(MODE_LIMITS["power"],(2,12))
        self.assertTrue(quota_reserve(a,"power",monotonic_now=1,utc_day="2026-11-15")[0])
        self.assertTrue(quota_reserve(a,"power",monotonic_now=2,utc_day="2026-11-15")[0])
        self.assertFalse(quota_reserve(a,"power",monotonic_now=3,utc_day="2026-11-15")[0])
        for i in range(10):
            self.assertTrue(quota_reserve(a,"power",monotonic_now=65+i*61,utc_day="2026-11-15")[0])
        self.assertFalse(quota_reserve(a,"power",monotonic_now=999,utc_day="2026-11-15")[0])

    def test_split_under_telegram_limit(self):
        text = "a"*7600
        chunks = split_telegram_text(text)
        self.assertTrue(all(len(c)<=3800 for c in chunks))
        self.assertEqual("".join(chunks),text)


if __name__ == "__main__":
    unittest.main()
