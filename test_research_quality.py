"""Offline safety and quality checks; no paid model or network calls."""
import unittest

from research_quality import (
    assess_answer, assess_sources, quality_header, requested_organizations,
    source_on_domains,
)


def verified(name, url, excerpt):
    return {"provider": name, "title": name, "url": url,
            "read_status": "verified_page_excerpt", "excerpt": excerpt}


class ResearchQualityTests(unittest.TestCase):
    def test_missing_ipcc_reports_partial_without_claiming_coverage(self):
        q = "Bandingkan NOAA, NASA dan IPCC tentang kenaikan paras laut."
        sources = [
            verified("NOAA", "https://www.climate.gov/sea-level", "Sea level rise."),
            verified("NASA", "https://sealevel.nasa.gov/faq", "Sea level rise."),
            {"url": "https://www.ipcc.ch/report/", "title": "IPCC",
             "read_status": "page_unavailable"},
        ]
        result = assess_sources(q, sources)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["missing"], ["IPCC"])
        self.assertIn("IPCC", quality_header(result))

    def test_no_verified_excerpts_is_insufficient_not_pass(self):
        result = assess_sources("Ollama local vs cloud", [
            {"url": "https://ollama.com/pricing", "read_status": "search_metadata_only"},
            {"url": "https://docs.ollama.com/cloud", "read_status": "page_unavailable"},
        ])
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")

    def test_citation_to_metadata_or_failed_page_is_flagged(self):
        sources = [
            verified("NOAA", "https://www.noaa.gov/sea", "Sea-level evidence 4.4 mm."),
            {"url": "https://www.ipcc.ch/report", "read_status": "page_unavailable"},
        ]
        result = assess_answer("NOAA vs IPCC", "NOAA: 4.4 mm [1]. IPCC [2].", sources)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["unsupported_citations"], [2])

    def test_amount_unsupported_by_excerpt_is_flagged(self):
        sources = [verified("Ollama", "https://ollama.com/pricing",
                            "Ollama Cloud Pro costs $20/month.")]
        result = assess_answer("Ollama pricing", "Plan costs US$500/month [1].", sources)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(result["unmatched_quantities"])

    def test_verified_amount_and_citation_pass_source_checks(self):
        sources = [verified("Ollama", "https://ollama.com/pricing",
                            "Ollama Cloud Pro costs $20/month.")]
        result = assess_answer("Ollama pricing", "Plan costs US$20/month [1].", sources)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["unsupported_citations"], [])
        self.assertEqual(result["unmatched_quantities"], [])

    def test_dont_confuse_fake_subdomain_with_official_host(self):
        self.assertFalse(source_on_domains(
            {"url": "https://ipcc.ch.attacker.example/post"}, ("ipcc.ch",)))
        self.assertTrue(source_on_domains(
            {"url": "https://www.ipcc.ch/report/"}, ("ipcc.ch",)))

    def test_known_named_organizations_are_explicit(self):
        requested = requested_organizations("NOAA versus NASA and IPCC")
        self.assertEqual(set(requested), {"NOAA", "NASA", "IPCC"})

    def test_unsupported_historical_year_is_flagged(self):
        s = [verified("NASA", "https://nasa.gov/sea", "Sea-level rise observed.")]
        result = assess_answer("NASA sea levels", "The report was written in 2025 [1].", s)
        self.assertIn("2025", result["unmatched_quantities"])
        self.assertEqual(result["status"], "PARTIAL")

    def test_historical_year_in_read_excerpt_passes(self):
        s = [verified("ASEAN", "https://asean.org/about",
                      "ASEAN was founded in Bangkok in 1967.")]
        result = assess_answer("Sejarah ASEAN", "ASEAN ditubuhkan pada 1967 [1].", s)
        self.assertEqual(result["status"], "PASS")

    def test_huge_citation_number_is_not_ignored(self):
        s = [verified("NASA", "https://nasa.gov/sea", "Sea-level rise observed.")]
        result = assess_answer("NASA sea levels", "NASA membincangkan paras laut [100].", s)
        self.assertEqual(result["unsupported_citations"], [100])
        self.assertEqual(result["status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
