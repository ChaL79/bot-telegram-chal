"""Offline regression tests for targeted recovery and source provenance."""
import unittest
from unittest.mock import patch

import httpx

from research_quality import assess_sources
from source_recovery import (
    _discovered_candidates, _scoped_query, recover_sources,
)


def ref(url, title, status, excerpt=None):
    out = {"url": url, "title": title, "snippet": title, "provider": "Test",
           "published": "", "read_status": status}
    if excerpt is not None:
        out["excerpt"] = excerpt
    return out


def page(text):
    return ("<html><main><article><h1>Scientific review</h1><p>" +
            text + "</p></article></main></html>").encode()


class SourceRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # CI safety regressions disable general reading globally. These tests
        # explicitly opt in under mocked transports, never the live network.
        self._reader_flag = patch.dict(
            "os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "1"})
        self._reader_flag.start()
        self.addCleanup(self._reader_flag.stop)

    async def test_missing_nasa_recovers_distinct_official_hostname(self):
        question = "Bandingkan NOAA NASA IPCC sea level rise projection uncertainty"
        existing = [
            ref("https://www.climate.gov/news/sea-level", "NOAA", "verified_page_excerpt",
                "Sea level rise from land ice and thermal expansion."),
            ref("https://science.nasa.gov/climate/sea-level", "NASA", "page_unavailable"),
            ref("https://www.ipcc.ch/report/ar6", "IPCC", "verified_page_excerpt",
                "IPCC sea level rise projection uncertainty scenarios and impacts."),
        ]
        discovery = [ref(
            "https://sealevel.nasa.gov/faq/12/what-causes-sea-level-rise/",
            "NASA sea level rise projection uncertainty", "search_metadata_only")]
        requests = []
        def mock(request):
            requests.append(str(request.url))
            return httpx.Response(200, headers={"content-type": "text/html"},
                content=page(
                    "NASA sea level rise projection uncertainty reflects ice sheet "
                    "changes and future global temperature scenarios, with a "
                    "likely range of coastal sea level changes."
                ))
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            with patch("source_recovery.public_source_search") as search:
                result = await recover_sources(question, discovery, existing, client=client)
        self.assertEqual(search.await_count, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(assess_sources(question, result)["status"], "PASS")
        self.assertEqual(result[2]["source_tier"], "official")
        self.assertEqual(result[2]["read_status"], "verified_page_excerpt")

    async def test_recovery_never_retries_failed_host(self):
        question = "NOAA NASA IPCC sea level rise"
        initial = [ref("https://www.ipcc.ch/report/", "IPCC", "page_unavailable"),
                   ref("https://www.noaa.gov/sea/level", "NOAA", "verified_page_excerpt",
                       "Sea level rise from global ocean warming.")]
        discovered = [ref("https://www.ipcc.ch/report/new/", "IPCC sea level rise",
                          "search_metadata_only")]
        def no_http(req):
            self.fail("Denied host should not be retried")
        async with httpx.AsyncClient(transport=httpx.MockTransport(no_http)) as client:
            with patch("source_recovery.public_source_search", return_value=[]) as search:
                result = await recover_sources(question, discovered, initial, client=client)
        self.assertEqual(len(result), len(initial))
        self.assertEqual(search.await_count, 1)

    async def test_secondary_evidence_does_not_claim_ipcc_coverage(self):
        question = "NOAA IPCC sea level rise projection"
        initial = [ref("https://www.noaa.gov/sea/level", "NOAA", "verified_page_excerpt",
                       "Sea level rise is influenced by warming."),
                   ref("https://www.ipcc.ch/report/", "IPCC", "page_unavailable")]
        discovered = [ref("https://example.org/science/ipcc-sea-level-rise-projection",
                          "Analysis of IPCC sea level rise projection", "search_metadata_only")]
        def mock(request):
            return httpx.Response(200, headers={"content-type": "text/html"},
                content=page("Independent analysis discusses IPCC sea level rise projection "
                             "scenarios and scientific uncertainties. Additional sea level "
                             "projection research considers polar ice changes."))
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            with patch("source_recovery.public_source_search", return_value=[]):
                result = await recover_sources(question, discovered, initial, client=client)
        self.assertEqual(assess_sources(question, result)["status"], "PARTIAL")
        self.assertTrue(any(s.get("source_tier") == "secondary" and
                            s.get("read_status") == "verified_page_excerpt"
                            for s in result))

    async def test_unrelated_or_unreadable_content_not_promoted(self):
        question = "NOAA IPCC sea level rise"
        initial = [ref("https://www.noaa.gov/sea/level", "NOAA", "verified_page_excerpt",
                       "Global sea level rise from warming."),
                   ref("https://www.ipcc.ch/report/", "IPCC", "page_unavailable")]
        discovered = [ref("https://example.org/ipcc/sea-level-rise",
                          "IPCC sea level rise", "search_metadata_only")]
        def mock(request):
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  content=page("This page discusses an unrelated subject "
                                               "and has no scientific measurements available."))
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            with patch("source_recovery.public_source_search", return_value=[]):
                result = await recover_sources(question, discovered, initial, client=client)
        self.assertFalse(any(s.get("read_status") == "verified_page_excerpt" and
                             s.get("source_tier") == "secondary" for s in result))
        self.assertEqual(assess_sources(question, result)["status"], "PARTIAL")

    async def test_only_two_additional_reads_no_unsafe_url(self):
        question = "IPCC sea level rise projection"
        initial = [ref("https://www.ipcc.ch/report/", "IPCC", "page_unavailable")]
        discovered = [
            ref(f"https://example{i}.org/science/ipcc-sea-level-rise-projection",
                "IPCC sea level rise projection", "search_metadata_only")
            for i in range(7)]
        discovered.append(ref("http://127.0.0.1/admin", "IPCC sea level rise",
                              "search_metadata_only"))
        count = []
        def mock(req):
            count.append(str(req.url))
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            with patch("source_recovery.public_source_search", return_value=[]):
                result = await recover_sources(question, discovered, initial, client=client)
        self.assertEqual(len(count), 2)
        self.assertTrue(all(url.startswith("https://example") for url in count))
        self.assertFalse(any(s.get("read_status") == "verified_page_excerpt" for s in result))

    async def test_disable_flag_prevents_all_recovery(self):
        initial = [ref("https://www.ipcc.ch/report", "IPCC", "page_unavailable")]
        with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "0"}):
            result = await recover_sources("IPCC sea level rise", [], initial)
        self.assertIs(result, initial)

    async def test_full_coverage_skips_network(self):
        initial = [ref("https://www.ipcc.ch/report", "IPCC", "verified_page_excerpt",
                       "IPCC sea level rise projections and uncertainty.")]
        with patch("source_recovery.public_source_search") as search:
            result = await recover_sources("IPCC sea level rise", [], initial)
        self.assertIs(result, initial)
        self.assertEqual(search.await_count, 0)

    def test_rejects_spoof_domain_and_keeps_topic_anchors(self):
        q = "IPCC sea level rise"
        discovered = [ref("https://ipcc.ch.evil.example.org/sea-level-rise",
                          "IPCC sea level rise", "search_metadata_only")]
        out = _discovered_candidates(q, discovered, [], ["IPCC"])
        self.assertEqual(out, [], "Lookalike domain must be rejected")
        self.assertIn("IPCC", _scoped_query(q, ["IPCC"]))


if __name__ == "__main__":
    unittest.main()
