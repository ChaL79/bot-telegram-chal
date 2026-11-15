import unittest

import httpx

from free_research import (
    allowed_source_url, parse_rss, parse_wikipedia, public_source_search,
    evidence_text, references_text, search_query_terms, source_relevance,
)


class FreeResearchTests(unittest.IsolatedAsyncioTestCase):
    def test_allow_public_https_only(self):
        self.assertTrue(allowed_source_url("https://example.org/article"))
        for url in ("http://example.org/", "https://localhost/x", "https://127.0.0.1/",
                    "https://192.168.1.1/x", "file:///etc/passwd",
                    "https://a:b@example.org/", "https://example.org/\nspoof"):
            self.assertFalse(allowed_source_url(url))

    def test_rss_parsing_and_safety(self):
        xml = b"""<rss><channel>
          <item><title>Official topic</title><link>https://example.org/official</link>
          <description>Short &amp; useful</description><pubDate>Sun, 15 Nov 2026</pubDate></item>
          <item><title>Unsafe</title><link>http://localhost/private</link></item>
        </channel></rss>"""
        found = parse_rss(xml, "Bing", 4)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["title"], "Official topic")
        self.assertIn("Short & useful", found[0]["snippet"])
        self.assertEqual(parse_rss(b"<html>blocked</html>", "Bing"), [])

    def test_wikipedia_formats_only_actual_search_results(self):
        found = parse_wikipedia({"query": {"search": [
            {"title": "Python (programming language)", "snippet": "An <b>interpreted</b> language"}
        ]}})
        self.assertEqual(len(found), 1)
        self.assertIn("Python_(programming_language)", found[0]["url"])
        self.assertNotIn("<b>", found[0]["snippet"])

    async def test_search_combines_sources_with_no_key(self):
        calls = []
        def mock(req):
            calls.append(str(req.url))
            if req.url.host == "www.bing.com":
                return httpx.Response(200, content=b"""<rss><channel><item><title>Python official</title>
                <link>https://docs.python.org/3/library/asyncio.html</link><description>Python async reference</description></item>
                </channel></rss>""")
            if req.url.host == "en.wikipedia.org":
                return httpx.Response(200, json={"query": {"search": [
                    {"title": "Python asyncio programming", "snippet": "Programming"}
                ]}})
            if req.url.host == "news.google.com":
                return httpx.Response(503)
            raise AssertionError("Unapproved host")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search("Cari sumber tentang Python asyncio rate limiting. Ringkaskan dapatan.", client=client)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(sources), 2)
        self.assertIn("docs.python.org", sources[0]["url"])
        self.assertIn("python+asyncio+rate+limiting", calls[0].lower())
        self.assertIn("metadata", references_text(sources))
        self.assertIn("[1]", evidence_text(sources))

    def test_extract_technical_topic_ignores_malay_command_words(self):
        query = "Cari sumber tentang Python asyncio rate limiting. Ringkaskan dapatan, bezakan fakta dan andaian dan sertakan pautan."
        self.assertEqual(search_query_terms(query), ["python", "asyncio", "rate", "limiting"])
        self.assertEqual(search_query_terms("/deepdive Cari Python asyncio rate limiting"), ["python", "asyncio", "rate", "limiting"])

    def test_irrelevant_cari_results_are_rejected(self):
        terms = search_query_terms("Cari sumber tentang Python asyncio rate limiting")
        irrelevant = {"title": "CARI | Consumer Aesthetics Research Institute", "url": "https://cari.institute/",
                      "snippet": "Aesthetic insights"}
        relevant = {"title": "Python asyncio rate limiting with queues", "url": "https://docs.example.org/asyncio/",
                    "snippet": "Use asyncio semaphore and token bucket rate limiter"}
        self.assertEqual(source_relevance(irrelevant, terms), 0)
        self.assertGreaterEqual(source_relevance(relevant, terms), 4)

    async def test_only_relevant_results_enter_research_context(self):
        received = []
        def mock(req):
            received.append(str(req.url))
            if req.url.host == "www.bing.com":
                return httpx.Response(200, content=b"""<rss><channel>
                    <item><title>CARI Network</title><link>https://cari.com/</link><description>Search a car</description></item>
                    <item><title>Python asyncio rate limiting techniques</title><link>https://dev.example.org/asyncio-limits</link>
                    <description>Use Python asyncio queues for request rate limiting</description></item></channel></rss>""")
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search("Cari sumber tentang Python asyncio rate limiting.", client=client)
        self.assertEqual(len(sources), 1)
        self.assertIn("asyncio", sources[0]["title"].lower())
        self.assertNotIn("cari.com", evidence_text(sources))
        self.assertIn("python+asyncio+rate+limiting", received[0].lower())

    async def test_provider_failure_fails_closed(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(403)
        )) as client:
            sources = await public_source_search("anything", client=client)
        self.assertEqual(sources, [])

    def test_evidence_metadata_only(self):
        src = [{"provider": "Bing RSS", "title": "Example", "url": "https://example.org/x",
                "snippet": "Not full article.", "published": ""}]
        self.assertIn("Snippet/headline only", evidence_text(src))
        self.assertIn("bukan semakan artikel penuh", references_text(src))


if __name__ == "__main__":
    unittest.main()
