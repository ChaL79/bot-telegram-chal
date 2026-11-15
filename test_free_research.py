import unittest

import httpx

from free_research import (
    allowed_source_url, parse_rss, parse_wikipedia, public_source_search,
    evidence_text, references_text, search_query_terms, source_relevance,
    parse_stackexchange, parse_hackernews,
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
            if req.url.host in ("api.stackexchange.com", "hn.algolia.com"):
                return httpx.Response(403)
            raise AssertionError("Unapproved host")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search("Cari sumber tentang Python asyncio rate limiting. Ringkaskan dapatan.", client=client)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(sources), 2)
        self.assertTrue(any("docs.python.org" in source["url"] for source in sources))
        self.assertIn("q=python%20asyncio%20rate%20limiting", calls[0].lower())
        self.assertIn("metadata", references_text(sources))
        self.assertIn("[1]", evidence_text(sources))

    def test_stackoverflow_metadata_parsing(self):
        data = {"items": [
            {"title": "Python asyncio rate limiting", "link": "https://stackoverflow.com/questions/1234/example",
             "tags": ["python", "python-asyncio", "rate-limiting"]},
            {"title": "Bad", "link": "http://localhost/private", "tags": ["python"]},
        ]}
        sources = parse_stackexchange(data)
        self.assertEqual(len(sources), 1)
        self.assertIn("asyncio", sources[0]["title"])
        self.assertIn("python-asyncio", sources[0]["snippet"])

    def test_hackernews_source_metadata(self):
        data = {"hits": [
            {"title": "Python asyncio rate limiting", "url": "https://example.org/article",
             "created_at": "2026-11-15"},
            {"title": "Local", "url": "http://localhost/secret"}
        ]}
        sources = parse_hackernews(data)
        self.assertEqual(len(sources), 1)
        self.assertIn("example.org", sources[0]["url"])

    async def test_technical_sources_survive_blocked_rss(self):
        hosts = []
        def mock(req):
            hosts.append(req.url.host)
            if req.url.host == "api.stackexchange.com":
                return httpx.Response(200, json={"items": [
                    {"title": "Python asyncio rate limiter with semaphore",
                     "link": "https://stackoverflow.com/questions/123/example",
                     "tags": ["python", "python-asyncio", "rate-limiting"]},
                ]})
            if req.url.host == "hn.algolia.com":
                return httpx.Response(200, json={"hits": [
                    {"title": "Python asyncio rate limiting using token bucket",
                     "url": "https://engineering.example.org/python-asyncio-rate"}
                ]})
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            items = await public_source_search(
                "Cari sumber tentang Python asyncio rate limiting.", client=client)
        self.assertIn("api.stackexchange.com", hosts)
        self.assertIn("hn.algolia.com", hosts)
        self.assertEqual(len(items), 2)
        self.assertTrue(all("asyncio" in x["title"].lower() for x in items))

    def test_aiolimiter_classified_as_technical(self):
        from free_research import technical_topic
        terms = search_query_terms(
            "Deep dive research whether aiolimiter's Leaky Bucket allows burst requests")
        self.assertIn("aiolimiter", terms)
        self.assertTrue(technical_topic(terms))

    def test_extract_technical_topic_ignores_malay_command_words(self):
        query = "Cari sumber tentang Python asyncio rate limiting. Ringkaskan dapatan, bezakan fakta dan andaian dan sertakan pautan."
        self.assertEqual(search_query_terms(query), ["python", "asyncio", "rate", "limiting"])
        self.assertEqual(search_query_terms("/deepdive Cari Python asyncio rate limiting"), ["python", "asyncio", "rate", "limiting"])

    def test_long_multisource_question_finds_security_subject_not_conduct(self):
        q = ("Conduct independent, multi-source cybersecurity research as of 15 November 2026. "
             "Compare three account authentication methods: FIDO2 passkeys, authenticator-app "
             "TOTP codes, and SMS one-time passwords. Research official publications from "
             "NIST, CISA and FIDO Alliance.")
        terms = search_query_terms(q)
        self.assertIn("fido2", terms)
        self.assertIn("passkeys", terms)
        self.assertIn("totp", terms)
        self.assertIn("sms", terms)
        self.assertNotIn("conduct", terms)
        self.assertNotIn("2026", terms)
        for title in ["Timeline of the Russo-Ukrainian war", "2025-2026 Iranian protests",
                      "conduct - English dictionary"]:
            self.assertEqual(source_relevance(
                {"title":title, "url":"https://en.wikipedia.org/wiki/Conduct",
                 "snippet":title}, terms), 0)
        self.assertGreater(source_relevance(
            {"title":"FIDO2 passkeys resist phishing attacks",
             "url":"https://fidoalliance.org/passkeys-2/",
             "snippet":"Passkeys based on FIDO2 authenticator security"}, terms), 0)

    def test_gemini_api_privacy_excludes_generic_google_and_launch_news(self):
        terms = search_query_terms(
            "Research Google Gemini API Free Tier privacy data use and training")
        for irrelevant in [
            {"title":"Google Maps - Wikipedia","url":"https://en.wikipedia.org/wiki/Google_Maps","snippet":"Maps navigation"},
            {"title":"Introducing Gemini 4 Hydrogen","url":"https://news.example/gemini-model","snippet":"New model launch"},
            {"title":"Google Drive","url":"https://en.wikipedia.org/wiki/Google_Drive","snippet":"Storage"}
        ]:
            self.assertEqual(source_relevance(irrelevant, terms), 0)
        policy = {"title":"Gemini API Additional Terms of Service - Unpaid Services",
                  "url":"https://ai.google.dev/gemini-api/terms",
                  "snippet":"Gemini API privacy, how Google uses data on unpaid and paid services"}
        self.assertGreater(source_relevance(policy, terms), 0)

    def test_aiolimiter_query_rejects_everify_and_va(self):
        terms = search_query_terms(
            "Verify from official aiolimiter documentation whether AsyncLimiter allows burst requests.")
        self.assertIn("aiolimiter", terms)
        self.assertNotIn("verify", terms)
        self.assertNotIn("official", terms)
        for item in [
            {"title": "Home | E-Verify", "url": "https://www.e-verify.gov/",
             "snippet": "Verify your eligibility and official records"},
            {"title": "Verify | Veterans Affairs", "url": "https://www.va.gov/verify/",
             "snippet": "Official government verification page"},
        ]:
            self.assertEqual(source_relevance(item, terms), 0)
        relevant = {"title": "aiolimiter AsyncLimiter burst rate limiting",
                    "url": "https://aiolimiter.readthedocs.io/en/stable/",
                    "snippet": "AsyncLimiter max_rate time_period permits burst requests"}
        self.assertGreater(source_relevance(relevant, terms), 0)

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
        self.assertIn("q=python%20asyncio%20rate%20limiting", received[0].lower())

    def test_science_query_requires_real_topic_and_verified_host(self):
        q = ("Compare NOAA NASA IPCC scientific explanations of global sea-level rise, "
             "thermal expansion of oceans and melting land ice.")
        terms = search_query_terms(q)
        self.assertTrue({"noaa", "nasa", "ipcc"}.issubset(set(terms)))
        self.assertTrue({"sea", "level"}.issubset(set(terms)))
        samples = [
            ("Latest Satellite Imagery", "https://www.nhc.noaa.gov/satellite.php",
             "hurricane imagery current storm"),
            ("20th century in science", "https://en.wikipedia.org/wiki/20th_century_in_science",
             "IPCC global climate century"),
            ("Scientific Consensus", "https://science.nasa.gov/climate-change/scientific-consensus/",
             "NASA scientific climate consensus"),
        ]
        for title, url, snippet in samples:
            self.assertEqual(source_relevance(
                {"title":title, "url":url, "snippet":snippet}, terms), 0)
        relevant = {"title":"Climate Change: Global Sea Level",
                    "url":"https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level",
                    "snippet":"NOAA ocean thermal expansion and glacier melting drive sea level rise"}
        self.assertGreater(source_relevance(relevant, terms), 0)

    async def test_institution_site_search_returns_organization_diverse_metadata(self):
        queries = []
        def mock(req):
            host = req.url.host
            if host == "www.bing.com":
                q = req.url.params.get("q", "")
                queries.append(q)
                if q.startswith("site:www.climate.gov"):
                    url = "https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level"
                    title = "NOAA Global Sea Level Thermal Expansion"
                elif q.startswith("site:sealevel.nasa.gov"):
                    url = "https://sealevel.nasa.gov/faq/12/what-causes-sea-level-rise/"
                    title = "NASA Sea Level Rise Thermal Expansion"
                elif q.startswith("site:www.ipcc.ch"):
                    url = "https://www.ipcc.ch/report/ar6/wg1/chapter/summary-for-policymakers/"
                    title = "IPCC Sea Level Rise Ocean Thermal Expansion"
                else:
                    return httpx.Response(200, content=b"<rss><channel></channel></rss>")
                xml = ("<rss><channel><item><title>" + title + "</title><link>" + url
                       + "</link><description>sea level rise ocean thermal expansion "
                       + "land ice melting</description></item></channel></rss>")
                return httpx.Response(200, content=xml.encode())
            if host == "en.wikipedia.org":
                return httpx.Response(200, json={"query":{"search":[]}})
            if host == "news.google.com":
                return httpx.Response(200, content=b"<rss><channel></channel></rss>")
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search(
                "Compare NOAA NASA IPCC global sea level rise thermal expansion land ice melting",
                client=client)
        self.assertEqual(len(queries), 4)
        self.assertEqual(len(sources), 3)
        self.assertTrue(all(item["provider"] == "Official site discovery via Bing RSS" for item in sources))
        self.assertIn("climate.gov", sources[0]["url"])
        self.assertIn("nasa.gov", sources[1]["url"])
        self.assertIn("ipcc.ch", sources[2]["url"])
        self.assertTrue(all(item.get("read_status") is None for item in sources))

    async def test_accessibility_discovery_uses_requested_organizations_only(self):
        queries = []
        def mock(req):
            if req.url.host == "www.bing.com":
                q = req.url.params.get("q", "")
                queries.append(q)
                host = ""
                if q.startswith("site:www.w3.org "):
                    host = "www.w3.org"
                    path = "/TR/WCAG22/"
                elif q.startswith("site:developer.mozilla.org "):
                    host = "developer.mozilla.org"
                    path = "/en-US/docs/Web/Accessibility/Keyboard"
                elif q.startswith("site:www.gov.uk "):
                    host = "www.gov.uk"
                    path = "/service-manual/helping-people-to-use-your-service/testing-for-accessibility"
                if host:
                    xml = ("<rss><channel><item><title>WCAG keyboard focus accessibility</title>"
                           "<link>https://" + host + path + "</link>"
                           "<description>WCAG 2.2 keyboard focus visibility testing</description>"
                           "</item></channel></rss>")
                    return httpx.Response(200, content=xml.encode())
                return httpx.Response(200, content=b"<rss><channel></channel></rss>")
            if req.url.host == "en.wikipedia.org":
                return httpx.Response(200, json={"query":{"search":[]}})
            if req.url.host == "news.google.com":
                return httpx.Response(200, content=b"<rss><channel></channel></rss>")
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            result = await public_source_search(
                "Compare W3C MDN Web Docs and GOV.UK WCAG 2.2 keyboard focus visibility testing",
                client=client)
        self.assertEqual(len(queries), 4, queries)
        self.assertEqual(len(result), 3, result)
        self.assertEqual(
            [httpx.URL(item["url"]).host for item in result],
            ["www.w3.org", "developer.mozilla.org", "www.gov.uk"])
        self.assertTrue(all(item.get("excerpt") is None for item in result))

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
