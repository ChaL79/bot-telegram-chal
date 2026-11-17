"""Offline cross-topic acceptance for general /deepdive.

Assertions are about real handler wiring, grounded page excerpts and honest
provenance. No network, API key or paid query is used by this test file.
"""
import json
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx

from free_research import (
    search_query_terms, parse_wikipedia, evidence_text, references_text,
)
from source_reader import enrich_with_read_excerpts
from public_article_reader import safe_public_article_url


class UniversalResearchTests(unittest.IsolatedAsyncioTestCase):
    def test_malay_general_query_terms_are_translated_without_full_translation(self):
        for question, required in (
            ("Kesan pencemaran alam sekitar terhadap kesihatan", {"effects", "pollution", "environment", "health"}),
            ("Sejarah pendidikan dan ekonomi di Brunei", {"history", "education", "economy"}),
            ("Kelebihan dan risiko senaman untuk kesihatan", {"risk", "exercise", "health"}),
            ("Perbandingan kos tenaga elektrik", {"cost", "energy", "electricity"}),
        ):
            terms = set(search_query_terms(question))
            with self.subTest(question=question):
                self.assertTrue(required.issubset(terms), terms)

    def test_malay_encyclopedia_is_explicitly_metadata_not_verified(self):
        payload = {"query": {"search": [{"title": "Sejarah Brunei",
                                         "snippet": "Brunei history summary"}]}}
        hits = parse_wikipedia(payload, language="ms")
        self.assertEqual(len(hits), 1)
        self.assertEqual(urlsplit(hits[0]["url"]).hostname, "ms.wikipedia.org")
        self.assertNotIn("read_status", hits[0])
        self.assertIn("SEARCH METADATA ONLY", evidence_text(hits))
        self.assertIn("Metadata sahaja", references_text(hits))
        self.assertEqual(parse_wikipedia(payload, language="invalid"), [])

    async def test_general_history_and_health_articles_are_read_by_default(self):
        # Three different article hostnames; the fourth remains metadata.
        for topic in ("sejarah Brunei", "kesihatan dan pemakanan"):
            refs = [
                {"provider": "Bing web-search RSS",
                 "title": topic + f" source {i}", "url": f"https://public{i}.example.org/articles/story{i}",
                 "snippet": topic, "published": ""}
                for i in range(1, 5)
            ]
            fetched = []
            async def html(url, *, client=None):
                fetched.append(url)
                return (f"<html><body><main><h1>{topic}</h1>"
                        f"<p>{topic} ialah topik penting dalam kajian "
                        "dan sumber yang berbeza menyediakan bukti relevan untuk analisis.</p>"
                        "</main></body></html>")
            # Production default reader must be ON. Explicit env=0 can disable.
            with patch.dict("os.environ", {}, clear=True):
                with patch("source_reader.read_public_html", side_effect=html):
                    result = await enrich_with_read_excerpts(topic, refs)
            self.assertEqual(len(fetched), 3)
            status = [r.get("read_status") for r in result]
            self.assertEqual(status.count("verified_page_excerpt"), 3)
            self.assertEqual(status.count("search_metadata_only"), 1)
            self.assertTrue(all(r.get("excerpt") for r in result[:3]))
            self.assertEqual(len({urlsplit(r["url"]).hostname for r in result[:3]}), 3)

    async def test_reader_can_be_disabled_without_deployment(self):
        refs = [{"provider": "RSS", "title": "School education research",
                 "url": "https://public.example.org/articles/education",
                 "snippet": "education school", "published": ""}]
        async def forbidden(url, *, client=None):
            raise AssertionError("general reading disabled")
        with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "0"}):
            with patch("source_reader.read_public_html", side_effect=forbidden):
                result = await enrich_with_read_excerpts("education school", refs)
        self.assertEqual(result[0]["read_status"], "search_metadata_only")

    async def test_failed_page_read_cannot_be_called_verified(self):
        refs = [{"provider": "RSS", "title": "Economic data and history",
                 "url": "https://public.example.org/articles/economy",
                 "snippet": "economy history", "published": ""}]
        async def failed(url, *, client=None):
            return ""
        with patch.dict("os.environ", {}, clear=True):
            with patch("source_reader.read_public_html", side_effect=failed):
                result = await enrich_with_read_excerpts("economy history", refs)
        self.assertEqual(result[0]["read_status"], "page_unavailable")
        self.assertNotIn("verified_page_excerpt", result[0])
        self.assertNotIn("Petikan halaman diambil", references_text(result))

    def test_security_url_filters_remain_strict(self):
        for url in (
            "http://example.org/post",
            "https://localhost/private",
            "https://127.0.0.1/path",
            "https://abc.example.org:8080/data",
            "https://public.example.org/admin",
            "https://public.example.org/a?token=secret",
            "https://public.example.org/download/data.pdf",
        ):
            with self.subTest(url=url):
                self.assertFalse(safe_public_article_url(url))
        self.assertTrue(safe_public_article_url("https://public.example.org/articles/research"))


if __name__ == "__main__":
    unittest.main()
