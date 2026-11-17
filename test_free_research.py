from urllib.parse import urlsplit
import unittest

import httpx

from free_research import (
    allowed_source_url, parse_rss, parse_wikipedia, public_source_search,
    evidence_text, references_text, search_query_terms, source_relevance,
    parse_stackexchange, parse_hackernews, parse_bing_news_rss,
    canonical_article_key,
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
            if req.url.host in ("ms.wikipedia.org", "news.google.com"):
                return httpx.Response(503)
            if req.url.host in ("api.stackexchange.com", "hn.algolia.com"):
                return httpx.Response(403)
            raise AssertionError("Unapproved host")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search("Cari sumber tentang Python asyncio rate limiting. Ringkaskan dapatan.", client=client)
        # One extra no-key Bing News discovery call is allowed when publishers
        # are missing; it must not change verified-document semantics.
        self.assertEqual(len(calls), 7)
        self.assertTrue(any("/news/search" in url for url in calls))
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

    def test_malay_fatigue_question_matches_english_official_articles(self):
        question = ("Deep research mengenai pengurusan keletihan bagi pekerja offshore "
                    "dan shift work. Cari dokumentasi rasmi CDC/NIOSH Amerika Syarikat "
                    "dan HSE United Kingdom. Terangkan risiko kekurangan tidur, "
                    "kerja syif malam, keselamatan pekerjaan dan langkah pencegahan.")
        terms = search_query_terms(question)
        self.assertEqual(terms[:3], ["fatigue", "shift", "work"])
        self.assertIn("cdc", terms)
        self.assertIn("hse", terms)
        self.assertIn("sleep", terms)
        items = [
            {"title": "Fatigue and Work - CDC/NIOSH",
             "url": "https://www.cdc.gov/niosh/fatigue/about/index.html",
             "snippet": "Fatigue from night shift work and shortened sleep affects worker safety"},
            {"title": "Fatigue - HSE",
             "url": "https://www.hse.gov.uk/humanfactors/topics/fatigue.htm",
             "snippet": "Fatigue in shift workers creates workplace safety risks"},
        ]
        for item in items:
            self.assertGreater(source_relevance(item, terms), 0, item["url"])
        unrelated = {"title":"Offshore company announcement",
                     "url":"https://example.org/press/offshore",
                     "snippet":"Company profile"}
        self.assertEqual(source_relevance(unrelated, terms), 0)

    def test_leo_geo_malay_query_keeps_orbits_ahead_of_generic_latency(self):
        q = ("/deepdive Bandingkan internet satelit LEO vs GEO untuk kapal offshore: "
             "latency, kelajuan, liputan, cuaca, dan handover.")
        terms = search_query_terms(q)
        self.assertEqual(terms[:3], ["satellite", "leo", "geo"])
        self.assertIn("offshore", terms)
        self.assertIn("maritime", terms)
        self.assertIn("speed", terms)
        self.assertIn("coverage", terms)
        self.assertIn("weather", terms)
        generic = {"title": "What is latency? - IBM",
                   "url": "https://www.ibm.com/think/topics/latency",
                   "snippet": "Network latency and bandwidth measurements"}
        self.assertEqual(source_relevance(generic, terms), 0)
        relevant = [
            {"title": "LEO GEO satellite broadband for maritime vessels",
             "url": "https://orbitalpha.org/guides/leo-geo-satellite-maritime",
             "snippet": "Compare offshore GEO LEO coverage and latency"},
            {"title": "LEO satellite handover for offshore connectivity",
             "url": "https://orbitbeta.net/articles/leo-satellite-handover",
             "snippet": "Maritime broadband weather and vessel handover"},
        ]
        for item in relevant:
            self.assertGreater(source_relevance(item, terms), 0)

    async def test_leo_geo_discovery_rejects_generic_latency_and_keeps_domains(self):
        q = ("Bandingkan internet satelit LEO dan GEO untuk kapal offshore, "
             "latency, kelajuan, liputan, cuaca dan handover.")
        rss = b"""<rss><channel>
        <item><title>What is latency? - IBM</title>
          <link>https://www.ibm.com/think/topics/latency</link>
          <description>Network latency and bandwidth</description></item>
        <item><title>LEO GEO satellite broadband for maritime vessels</title>
          <link>https://orbitalpha.org/guides/leo-geo-satellite-maritime</link>
          <description>GEO and LEO offshore coverage latency</description></item>
        <item><title>LEO satellite handover for offshore connectivity</title>
          <link>https://orbitbeta.net/articles/leo-satellite-handover</link>
          <description>Maritime vessels weather and handover</description></item>
        </channel></rss>"""
        def mock(request):
            if request.url.host == "www.bing.com":
                return httpx.Response(200, content=rss)
            if request.url.host == "en.wikipedia.org":
                return httpx.Response(200, json={"query": {"search": []}})
            return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            results = await public_source_search(q, client=client)
        hosts = {urlsplit(r["url"]).hostname for r in results}
        self.assertEqual(hosts, {"orbitalpha.org", "orbitbeta.net"})
        self.assertNotIn("www.ibm.com", hosts)

    def test_bing_news_publisher_link_extraction_is_strict_and_no_fetch(self):
        from urllib.parse import quote
        from xml.sax.saxutils import escape
        good = "https://publisher-one.org/articles/satellite-leo-geo"
        safe = "http://www.bing.com/news/apiclick.aspx?ref=FexRss&url=" + quote(good, safe="")
        unsafe = [
            "http://www.bing.com/news/apiclick.aspx?url=" + quote("http://127.0.0.1/admin", safe=""),
            "http://www.bing.com/news/apiclick.aspx?url=" + quote("https://localhost/a", safe=""),
            "http://evil.com/news/apiclick.aspx?url=" + quote(good, safe=""),
            "http://www.bing.com/redirect?url=" + quote(good, safe=""),
            "http://www.bing.com/news/apiclick.aspx?url=" + quote(good + "?tracking=1", safe=""),
            "http://www.bing.com/news/apiclick.aspx?url=" + quote(good, safe="") +
            "&url=" + quote("https://evil.org/article", safe=""),
        ]
        items = [(safe, "LEO GEO satellite maritime broadband")] + [
            (u, "Untrusted") for u in unsafe]
        xml = ("<rss><channel>" + "".join(
            "<item><title>" + title + "</title><link>" + escape(url) +
            "</link><description>LEO GEO satellite internet</description></item>"
            for url, title in items) + "</channel></rss>").encode()
        results = parse_bing_news_rss(xml)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], good)
        self.assertIn("not yet verified", results[0]["provider"])
        self.assertEqual(parse_bing_news_rss(b"<!DOCTYPE rss><rss/>"), [])

    def test_amp_alias_is_same_article_not_same_as_unrelated_hosts(self):
        a = "https://publisher.example.org/amp/news/story"
        b = "https://publisher.example.org/news/story"
        self.assertEqual(canonical_article_key(a), canonical_article_key(b))
        self.assertNotEqual(canonical_article_key(b + "?x=2"), canonical_article_key(b))
        self.assertNotEqual(
            canonical_article_key("https://other.publisher.example.org/news/story"),
            canonical_article_key(b))

    async def test_amp_alias_not_counted_twice_in_universal_discovery(self):
        rss = b"""<rss><channel>
        <item><title>LEO GEO satellite maritime vessel communications</title>
        <link>https://publisher.example.org/amp/news/leo-satellite</link>
        <description>Offshore satellite low earth orbit</description></item>
        <item><title>LEO GEO satellite maritime vessel communications</title>
        <link>https://publisher.example.org/news/leo-satellite</link>
        <description>Offshore satellite low earth orbit</description></item>
        <item><title>LEO GEO satellite offshore coverage handover</title>
        <link>https://second.example.net/article/leo-geo-maritime</link>
        <description>Satellite offshore communications</description></item>
        </channel></rss>"""
        def mock(req):
            if req.url.host == "www.bing.com" and req.url.path == "/search":
                return httpx.Response(200, content=rss)
            if req.url.host == "en.wikipedia.org":
                return httpx.Response(200, json={"query": {"search": []}})
            return httpx.Response(404)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            out = await public_source_search(
                "LEO GEO satellite maritime vessel coverage handover", client=client)
        self.assertEqual(len({canonical_article_key(x["url"]) for x in out}), len(out))
        self.assertEqual(len(out), 2)

    async def test_news_fallback_recovers_independent_satellite_publishers(self):
        from urllib.parse import quote
        from xml.sax.saxutils import escape
        articles = [
            ("LEO GEO satellite for maritime vessels",
             "https://publisher-alpha.org/insight/leo-geo-satellite"),
            ("LEO satellite connectivity for offshore ships",
             "https://publisher-beta.net/research/leo-satellite-maritime"),
        ]
        rss = ("<rss><channel>" + "".join(
            "<item><title>" + title + "</title><link>" +
            escape("http://www.bing.com/news/apiclick.aspx?url=" +
                   quote(url, safe="")) +
            "</link><description>Satellite LEO GEO offshore coverage handover</description></item>"
            for title, url in articles) + "</channel></rss>").encode()
        calls = []
        def mock(req):
            calls.append((req.url.host, req.url.path))
            if req.url.host == "www.bing.com" and req.url.path == "/news/search":
                return httpx.Response(200, content=rss)
            if req.url.host == "www.bing.com":
                return httpx.Response(200, content=b"<rss><channel/></rss>")
            if req.url.host == "en.wikipedia.org":
                return httpx.Response(200, json={"query":{"search":[]}})
            return httpx.Response(503)
        question = "Bandingkan internet satelit LEO GEO kapal offshore latency cuaca handover"
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search(question, client=client)
        self.assertEqual({urlsplit(x["url"]).hostname for x in sources},
                         {"publisher-alpha.org", "publisher-beta.net"})
        self.assertGreaterEqual(sum(path == "/news/search" for host, path in calls), 1)
        self.assertLessEqual(sum(path == "/news/search" for host, path in calls), 2)

    def test_wifi_research_keeps_topic_not_malay_instructions(self):
        question = ("/deepdive Deep research: Bandingkan Wi-Fi 7 dengan Wi-Fi 6 berdasarkan "
                    "sumber rasmi Wi-Fi Alliance, Intel dan TP-Link. Semak perbezaan kelajuan, "
                    "latency, jalur 6 GHz, Multi-Link Operation (MLO), dan manfaat sebenar "
                    "untuk internet rumah 500 Mbps. Cuba baca kandungan HTML sebenar daripada "
                    "sekurang-kurangnya dua domain berbeza. Sertakan URL, petikan sebenar "
                    "setiap halaman yang berjaya dibaca, dan asingkan sumber metadata sahaja "
                    "atau gagal diakses. Jangan reka fakta, petikan atau keputusan ujian.")
        terms = search_query_terms(question)
        self.assertEqual(terms[:3], ["wifi7", "wifi", "wifi6"])
        self.assertTrue({"mlo", "6ghz", "latency"}.issubset(set(terms)))
        self.assertNotIn("bandingkan", terms)
        self.assertNotIn("berdasarkan", terms)
        self.assertNotIn("sekurang", terms)
        self.assertNotIn("petikan", terms)

    def test_wifi_version_matching_accepts_real_articles_not_generic_wifi(self):
        terms = search_query_terms("Wi-Fi 7 vs Wi-Fi 6 MLO 6 GHz")
        for item in [
            {"title": "Wi-Fi 7 multi-link operation MLO explained",
             "url": "https://example.org/guides/wifi-7-mlo/",
             "snippet": "Wi-Fi 7 technology and 6 GHz"},
            {"title": "Wi-Fi 6 compared to Wi-Fi 7",
             "url": "https://example.org/articles/wifi6-vs-wifi7",
             "snippet": "Wireless networking standards"},
        ]:
            self.assertGreater(source_relevance(item, terms), 0)
        generic = {"title": "Public Wi-Fi hotspots", "url": "https://example.org/hotspots",
                   "snippet": "Free internet services"}
        self.assertEqual(source_relevance(generic, terms), 0)

    async def test_wifi_search_accepts_source_results_after_normalization(self):
        question = "Bandingkan Wi-Fi 7 dengan Wi-Fi 6, latency dan Multi-Link Operation."
        received = []
        def mock(req):
            if req.url.host == "www.bing.com":
                received.append(str(req.url.params.get("q")))
                return httpx.Response(200, content=b"""<rss><channel>
                    <item><title>Wi-Fi 7 multi-link operation MLO</title>
                    <link>https://example.org/wifi7-mlo</link>
                    <description>Compare Wi-Fi 7 and Wi-Fi 6 and latency improvements</description></item>
                    <item><title>Unrelated shopping news</title>
                    <link>https://example.org/shopping</link>
                    <description>Nothing about networking</description></item>
                </channel></rss>""")
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            result = await public_source_search(question, client=client)
        self.assertEqual(len(result), 1)
        self.assertIn("Wi-Fi 7", result[0]["title"])
        self.assertIn("wifi7", received[0])

    def test_direct_article_urls_rank_above_news_redirects_and_marketing(self):
        from free_research import prioritize_readable_sources, readable_article_priority
        refs = [
            {"url": "https://news.google.com/rss/articles/article?oc=5",
             "title": "Research news index"},
            {"url": "https://www.tp-link.com/us/wifi7/",
             "title": "WiFi 7 marketing landing page"},
            {"url": "https://www.intel.com/content/www/us/en/products/details/wireless/wi-fi-7-series.html",
             "title": "Intel Wi-Fi 7 reference"},
            {"url": "https://www.spectrum.com/resources/internet-wifi/what-is-wifi-7",
             "title": "Spectrum Wi-Fi 7 article"},
            {"url": "https://www.highspeedinternet.com/resources/what-is-wi-fi-7",
             "title": "Wi-Fi 7 article"},
        ]
        ranked = prioritize_readable_sources([(10, r) for r in refs])
        chosen = [x["url"] for _score, x in ranked[:3]]
        self.assertEqual(set(chosen), {refs[2]["url"], refs[3]["url"], refs[4]["url"]})
        self.assertEqual(readable_article_priority(refs[0]), 0)
        self.assertGreater(readable_article_priority(refs[2]),
                           readable_article_priority(refs[1]))

    def test_priority_does_not_bypass_safe_url_checks(self):
        from free_research import readable_article_priority
        for url in ("http://www.intel.com/en/articles/wifi7",
                    "https://127.0.0.1/resource/wifi7.html",
                    "https://example.org/docs/article?api_key=secret",
                    "https://evil.local/resources/wifi7.html",
                    "https://example.org/resources/../../private"):
            self.assertEqual(readable_article_priority({"url": url}), 0)

    async def test_ssd_brand_scoped_discovery_and_selection(self):
        from free_research import requested_product_institutions
        question = ("Bandingkan SSD NVMe PCIe dengan SSD SATA berdasarkan "
                    "dokumentasi rasmi Kingston, Crucial dan Samsung.")
        self.assertEqual(requested_product_institutions(question),
                         ["kingston", "crucial", "samsung"])
        seen = []
        def mock(req):
            query = str(req.url.params.get("q", ""))
            seen.append(query)
            if "kingston" in query.split() and not query.startswith("site:"):
                return httpx.Response(200, content=b"""<rss><channel><item>
                  <title>NVMe vs SATA SSDs Kingston</title>
                  <link>https://www.kingston.com/en/blog/pc-performance/nvme-vs-sata</link>
                  <description>SSD NVMe PCIe performance versus SATA</description>
                  </item><item><title>Fake offsite Kingston SSD</title>
                  <link>https://evil.example/kingston/ssd</link>
                  <description>NVMe SATA</description></item></channel></rss>""")
            if "crucial" in query.split() and not query.startswith("site:"):
                return httpx.Response(200, content=b"""<rss><channel><item>
                  <title>Crucial SSD SATA and NVMe</title>
                  <link>https://www.crucial.com/articles/about-ssd/nvme-vs-sata</link>
                  <description>NVMe PCIe and SATA SSD interface</description>
                  </item></channel></rss>""")
            if "samsung" in query.split() and not query.startswith("site:"):
                return httpx.Response(200, content=b"""<rss><channel><item>
                  <title>Samsung NVMe SSD SATA product guide</title>
                  <link>https://semiconductor.samsung.com/consumer-storage/support/faqs/internalssd-product-information/</link>
                  <description>NVMe PCIe SSD SATA performance</description>
                  </item></channel></rss>""")
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            found = await public_source_search(question, client=client)
        self.assertEqual(len([q for q in seen if any(org in q.split() for org in ("kingston", "crucial", "samsung"))]), 3)
        hosts = {httpx.URL(item["url"]).host for item in found[:3]}
        self.assertEqual(hosts, {"www.kingston.com", "www.crucial.com",
                                 "semiconductor.samsung.com"})
        self.assertFalse(any("evil.example" in item["url"] for item in found))

    async def test_official_sitemap_discovery_is_url_metadata_only(self):
        question = ("Bandingkan SSD NVMe PCIe dengan SSD SATA dokumentasi "
                    "Kingston Crucial Samsung")
        sitemap = b"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://semiconductor.samsung.com/ssd/pc-ssd/pm9e1/</loc></url>
          <url><loc>https://semiconductor.samsung.com/consumer-storage/support/faqs/internalssd-product-information/</loc></url>
          <url><loc>https://127.0.0.1/ssd/private</loc></url>
          <url><loc>https://evil.example/ssd/nvme/</loc></url>
          <url><loc>https://semiconductor.samsung.com/account/ssd/</loc></url>
          </urlset>"""
        def mock(req):
            if str(req.url) == "https://semiconductor.samsung.com/sitemap.xml":
                return httpx.Response(200, headers={"content-type":"application/xml"},content=sitemap)
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            sources = await public_source_search(question,client=client)
        samsung = [item for item in sources if item["url"].startswith("https://semiconductor.samsung.com")]
        self.assertGreaterEqual(len(samsung),1)
        self.assertEqual(samsung[0]["url"],
            "https://semiconductor.samsung.com/consumer-storage/support/faqs/internalssd-product-information/")
        self.assertIn("sitemap",samsung[0]["provider"].lower())
        self.assertFalse(any("evil.example" in x["url"] or "127.0.0.1" in x["url"] for x in sources))
        self.assertTrue(all("excerpt" not in x for x in sources))

    async def test_official_multi_brand_research_excludes_retail_offers(self):
        from free_research import is_marketplace_listing
        self.assertTrue(is_marketplace_listing(
            "https://computers.woot.com/offers/samsung-ssd"))
        self.assertFalse(is_marketplace_listing(
            "https://semiconductor.samsung.com/consumer-storage/support/faqs/internalssd-product-information/"))
        rss = b"""<rss><channel><item><title>Samsung 960 EVO NVMe SATA SSD - Woot</title>
                  <link>https://www.woot.com/offers/samsung-960-evo-nvme-m-2-ssd-1tb-0</link>
                  <description>NVMe SATA SSD</description></item></channel></rss>"""
        question = ("Bandingkan SSD NVMe PCIe dengan SSD SATA berdasarkan "
                    "dokumentasi rasmi Kingston Crucial Samsung.")
        def mock(req):
            if req.url.host == "www.bing.com":
                return httpx.Response(200,content=rss)
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            found = await public_source_search(question,client=client)
        self.assertFalse(any("woot.com" in x["url"] for x in found))

    def test_brand_scoping_not_used_for_unrelated_questions(self):
        from free_research import requested_product_institutions
        self.assertEqual(requested_product_institutions("Python asyncio semaphore"), [])
        self.assertEqual(requested_product_institutions("Kingston SSD specs"), ["kingston"])

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
