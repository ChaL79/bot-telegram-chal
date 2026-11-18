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
            if req.url.host == "docs.python.org":
                return httpx.Response(200, headers={"Content-Type": "text/html"},
                    content=b"<html><main><p>Python reference documentation describes Python programming and language syntax.</p></main></html>")
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

    def test_deterministic_quotes_are_exact_and_source_bound(self):
        from gemini_advanced import verified_source_quotes, suppress_model_quotes
        sources = [
            {"title": "IBM documentation", "url": "https://www.ibm.com/think/topics/nvme",
             "read_status": "verified_page_excerpt",
             "excerpt": "IBM NVMe allows SSDs to connect through PCIe and reduce traditional storage overhead."},
            {"title": "Samsung documentation", "url": "https://semiconductor.samsung.com/ssd/",
             "read_status": "verified_page_excerpt",
             "excerpt": "Samsung NVMe SSDs with PCIe are much faster than SATA SSDs for transfers."},
            {"title": "Kingston", "url": "https://www.kingston.com/en/ssd/",
             "read_status": "page_unavailable", "excerpt": "FAKE metadata quote must not appear."},
        ]
        output = verified_source_quotes(sources, limit_words=20)
        self.assertIn('Petikan disahkan: "IBM NVMe allows SSDs', output)
        self.assertIn('Petikan disahkan: "Samsung NVMe SSDs', output)
        self.assertNotIn('FAKE metadata quote', output)
        self.assertEqual(output.count("Petikan disahkan:"), 2)
        self.assertLess(output.index("www.ibm.com"), output.index("Samsung NVMe SSDs"))
        bad = ('IBM: "Samsung NVMe SSDs with PCIe are much faster than SATA SSDs for transfers."\n'
               'Samsung: "Samsung NVMe SSDs with PCIe are much faster than SATA SSDs for transfers."')
        cleaned = suppress_model_quotes(bad)
        self.assertNotIn("Samsung NVMe SSDs with PCIe", cleaned)
        self.assertIn("IBM:", cleaned)

    def test_prefers_real_body_quote_over_identical_article_title(self):
        from gemini_advanced import verified_source_quotes, RESEARCH_PROMPT
        source = [{
            "title": "LEO Satellite Networks: Supporting Maritime Safety",
            "url": "https://publisher.example.org/amp/news/leo-satellite",
            "read_status": "verified_page_excerpt",
            "excerpt": ("LEO Satellite Networks: Supporting Maritime Safety\n"
                        "Satellite networks in low Earth orbit support faster connections "
                        "for offshore maritime vessels."),
        }]
        output = verified_source_quotes(source)
        self.assertIn('Petikan disahkan: "Satellite networks in low Earth orbit', output)
        self.assertNotIn('Petikan disahkan: "LEO Satellite Networks:', output)
        self.assertIn("SEMUA aspek", RESEARCH_PROMPT)
        self.assertIn("belum disahkan", RESEARCH_PROMPT)

    def test_only_article_title_is_not_misrepresented_as_body_quote(self):
        from gemini_advanced import verified_source_quotes
        source = [{
            "title": "Satellite internet connectivity for offshore maritime vessels",
            "url": "https://publisher.example.org/article",
            "read_status": "verified_page_excerpt",
            "excerpt": "Satellite internet connectivity for offshore maritime vessels",
        }]
        self.assertEqual(verified_source_quotes(source), "")

    async def test_deepdive_replaces_model_misattributed_quote_with_verified_source(self):
        from unittest.mock import patch
        sources = [
            {"title": "IBM NVMe", "provider": "Reviewed secondary technical documentation",
             "url": "https://www.ibm.com/think/topics/nvme",
             "read_status": "verified_page_excerpt", "published": "",
             "excerpt": "IBM NVMe connects high-speed SSDs through the PCIe interface rather than using older SATA technology."},
            {"title": "Samsung SSD", "provider": "Official manufacturer sitemap",
             "url": "https://semiconductor.samsung.com/consumer-storage/support/faqs/internalssd-product-information/",
             "read_status": "verified_page_excerpt", "published": "",
             "excerpt": "Samsung NVMe SSDs with PCIe are much faster than SATA SSDs in many workloads."},
        ]
        fake_answer = ('IBM: "Samsung NVMe SSDs with PCIe are much faster than SATA SSDs in many workloads."\n'
                       "Kedua-dua sumber membandingkan teknologi storan.")
        def ai_mock(req):
            return httpx.Response(200, json={"candidates":[{"content":{"parts":[{"text":fake_answer}]}}]})
        with patch("gemini_advanced.public_source_search", return_value=sources):
            with patch("gemini_advanced.enrich_with_read_excerpts", return_value=sources):
                async with httpx.AsyncClient(transport=httpx.MockTransport(ai_mock)) as client:
                    result, ok = await run_advanced(
                        "Bandingkan SSD NVMe SATA", "TEST_KEY", "deepdive", client=client)
        self.assertTrue(ok)
        self.assertEqual(result.count("Samsung NVMe SSDs with PCIe are much faster"), 1)
        self.assertIn('Petikan disahkan: "IBM NVMe connects', result)
        self.assertIn('Petikan disahkan: "Samsung NVMe SSDs', result)
        self.assertIn("https://www.ibm.com/think/topics/nvme", result)
        self.assertIn("https://semiconductor.samsung.com", result)

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
            if req.url.host == "docs.python.org":
                return httpx.Response(200, headers={"Content-Type": "text/html"},
                    content=b"<html><main><p>Python asyncio semaphore synchronization manages concurrent tasks using asynchronous primitives.</p></main></html>")
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
        self.assertEqual(GEMINI_READ_TIMEOUT["deepdive"], 85.0)
        self.assertEqual(GEMINI_READ_TIMEOUT["power"], 70.0)

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

    async def test_power_uses_high_thinking_on_free_flash_lite(self):
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
        self.assertEqual(seen[0][1]["generationConfig"]["thinkingConfig"],
                         {"thinkingLevel": "high", "includeThoughts": False})
        self.assertEqual(seen[0][1]["generationConfig"]["maxOutputTokens"], 6144)
        self.assertNotIn("tools", seen[0][1])

    async def test_deepdive_uses_high_thinking_and_bounded_output(self):
        from unittest.mock import patch
        seen = []
        refs = [{"title": "NASA research", "provider": "Official research",
                 "url": "https://science.nasa.gov/climate-change/",
                 "read_status": "verified_page_excerpt",
                 "excerpt": "NASA scientific research measures and explains climate trends.",
                 "published": ""}]
        def mock(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200, json={"candidates": [
                {"content": {"parts": [{"text": "Dapatan berdasarkan bukti."}]}}
            ]})
        with patch("gemini_advanced.public_source_search", return_value=refs):
            with patch("gemini_advanced.enrich_with_read_excerpts", return_value=refs):
                async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
                    answer, ok = await run_advanced(
                        "Jelaskan bukti perubahan iklim", "TEST_KEY",
                        "deepdive", client=cli)
        self.assertTrue(ok)
        self.assertIn("Dapatan berdasarkan bukti", answer)
        self.assertEqual(len(seen), 1)
        config = seen[0]["generationConfig"]
        self.assertEqual(config["thinkingConfig"],
                         {"thinkingLevel": "high", "includeThoughts": False})
        self.assertEqual(config["maxOutputTokens"], 8192)
        self.assertNotIn("tools", seen[0])

    async def test_high_thinking_hidden_thoughts_not_in_answer(self):
        from unittest.mock import patch
        refs = [{"title": "Public study", "provider": "Open web",
                 "url": "https://example.org/study",
                 "read_status": "verified_page_excerpt",
                 "excerpt": "This public study examines evidence and science.", "published": ""}]
        def mock(req):
            return httpx.Response(200, json={"candidates": [
                {"content": {"parts": [
                    {"text": "Hidden thinking token", "thought": True},
                    {"text": "Visible evidence summary."}
                ]}}
            ]})
        with patch("gemini_advanced.public_source_search", return_value=refs):
            with patch("gemini_advanced.enrich_with_read_excerpts", return_value=refs):
                async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
                    answer, ok = await run_advanced(
                        "Sila kaji", "TEST_KEY", "deepdive", client=cli)
        self.assertTrue(ok)
        self.assertIn("Visible evidence summary", answer)
        self.assertNotIn("Hidden thinking token", answer)

    async def test_max_tokens_without_visible_answer_is_reported_without_retry(self):
        called = []
        def mock(req):
            called.append(req)
            return httpx.Response(200, json={
                "candidates": [{"finishReason": "MAX_TOKENS",
                                "content": {"parts": [{"text": "internal thought",
                                                       "thought": True}]}}]
            })
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            answer, ok = await run_advanced("Audit", "TEST_KEY", "power", client=cli)
        self.assertFalse(ok)
        self.assertIn("had token", answer)
        self.assertEqual(len(called), 1)

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
        self.assertIn("tidak akan meneka", answer)

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
