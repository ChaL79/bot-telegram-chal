"""Strict scoped release gate tests: verified secondary authority is not primary coverage."""
import unittest
from research_release_gate import acceptance_findings


def source(host, tier="initial", scope="", ipcc=False, asean=False,
           status="verified_page_excerpt"):
    return {"host": host, "status": status, "tier": tier, "scope": scope,
            "supports_ipcc_ar6_sea_level": ipcc,
            "supports_asean_founding_1967": asean}


def scoped():
    return [
        {"case": "NOAA_NASA_IPCC", "status": "PARTIAL",
         "missing_requested": ["IPCC"], "final_verified": 3,
         "final": [
             source("www.climate.gov"), source("sealevel.nasa.gov"),
             source("sealevel.nasa.gov", tier="delegated_official",
                    scope="IPCC_AR6_NASA_DATA_ONLY", ipcc=True)]},
        {"case": "OLLAMA_LOCAL_CLOUD", "status": "PASS",
         "missing_requested": [], "final_verified": 2,
         "final": [source("docs.ollama.com"), source("ollama.com")]},
        {"case": "GENERAL_HISTORY", "status": "PARTIAL",
         "missing_requested": ["ASEAN"], "final_verified": 2,
         "final": [
             source("asean.mfa.go.th", tier="founding_government", asean=True),
             source("www.mfa.gov.sg", tier="founding_government", asean=True)]},
    ]


class ReleaseGateTests(unittest.TestCase):
    def test_scoped_independently_corrobated_evidence_passes(self):
        self.assertEqual(acceptance_findings(scoped()), [])

    def test_one_founding_government_is_insufficient(self):
        reports = scoped()
        reports[2]["final"].pop()
        self.assertTrue(any("ASEAN" in msg
                            for msg in acceptance_findings(reports)))

    def test_nasa_ipcc_claim_must_be_verified_from_excerpt(self):
        reports = scoped()
        reports[0]["final"][-1]["supports_ipcc_ar6_sea_level"] = False
        self.assertTrue(any("IPCC" in msg
                            for msg in acceptance_findings(reports)))

    def test_unrelated_nasa_page_cannot_claim_ipcc_ar6(self):
        reports = scoped()
        reports[0]["final"][-1]["scope"] = "NASA_GENERAL"
        self.assertTrue(any("IPCC" in msg
                            for msg in acceptance_findings(reports)))

    def test_ipcc_original_domain_missing_remains_partial(self):
        reports = scoped()
        reports[0]["status"] = "PASS"
        self.assertTrue(any("Climate" in msg
                            for msg in acceptance_findings(reports)))

    def test_asean_original_domain_missing_remains_partial(self):
        reports = scoped()
        reports[2]["status"] = "PASS"
        self.assertTrue(any("ASEAN" in msg
                            for msg in acceptance_findings(reports)))

    def test_no_delegated_or_primary_ipcc_fails(self):
        reports = scoped()
        reports[0]["final"][-1]["status"] = "page_unavailable"
        self.assertTrue(any("IPCC" in msg
                            for msg in acceptance_findings(reports)))

    def test_original_official_sources_can_pass_full_coverage(self):
        reports = scoped()
        reports[0]["status"] = "PASS"
        reports[0]["missing_requested"] = []
        reports[0]["final"][-1] = source("www.ipcc.ch")
        reports[2]["status"] = "PASS"
        reports[2]["missing_requested"] = []
        reports[2]["final"] = [source("asean.org")]
        self.assertEqual(acceptance_findings(reports), [])

    def test_ollama_official_docs_and_pricing_both_required(self):
        reports = scoped()
        reports[1]["final"] = [source("docs.ollama.com"),
                               source("docs.ollama.com")]
        self.assertTrue(any("Ollama" in msg
                            for msg in acceptance_findings(reports)))

    def test_spoofed_institution_host_fails(self):
        reports = scoped()
        reports[2]["final"][0]["host"] = "asean.mfa.go.th.attacker.example"
        self.assertTrue(any("ASEAN" in msg
                            for msg in acceptance_findings(reports)))

    def test_metadata_cannot_count_as_verified(self):
        reports = scoped()
        reports[2]["final"][0]["status"] = "search_metadata_only"
        self.assertTrue(any("ASEAN" in msg
                            for msg in acceptance_findings(reports)))

    def test_missing_cases_fail_closed(self):
        self.assertTrue(acceptance_findings([]))
        self.assertTrue(acceptance_findings(scoped()[:1]))


if __name__ == "__main__":
    unittest.main()
