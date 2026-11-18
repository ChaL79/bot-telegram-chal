"""Release gate must fail closed even when a diagnostic job is green."""
import unittest

from research_release_gate import acceptance_findings


def reference(host, status="verified_page_excerpt", tier="initial"):
    return {"host": host, "status": status, "tier": tier}


def valid():
    return [
        {"case": "NOAA_NASA_IPCC", "status": "PASS", "final_verified": 3,
         "final": [reference("www.climate.gov"),
                   reference("sealevel.nasa.gov"),
                   reference("www.ipcc.ch")]},
        {"case": "OLLAMA_LOCAL_CLOUD", "status": "PASS", "final_verified": 2,
         "final": [reference("docs.ollama.com"), reference("ollama.com")]},
        {"case": "GENERAL_HISTORY", "status": "PASS", "final_verified": 2,
         "final": [reference("asean.org"), reference("www.mfa.gov.sg")]},
    ]


class ReleaseGateTests(unittest.TestCase):
    def test_all_verified_official_sources_pass(self):
        self.assertEqual(acceptance_findings(valid()), [])

    def test_missing_ipcc_primary_source_fails_even_with_delegated_ddc(self):
        reports = valid()
        reports[0]["status"] = "PARTIAL"
        reports[0]["final"][-1] = reference(
            "ipcc-data.climate.columbia.edu", tier="delegated_official")
        findings = acceptance_findings(reports)
        self.assertTrue(any("IPCC" in item for item in findings))

    def test_asean_metadata_only_fails(self):
        reports = valid()
        reports[2]["final"][0]["status"] = "search_metadata_only"
        self.assertTrue(any("ASEAN" in item for item
                            in acceptance_findings(reports)))

    def test_ollama_single_excerpt_fails(self):
        reports = valid()
        reports[1]["final_verified"] = 1
        reports[1]["final"][1]["status"] = "page_unavailable"
        self.assertTrue(any("Ollama" in item for item
                            in acceptance_findings(reports)))

    def test_missing_and_empty_reports_fail_closed(self):
        self.assertGreater(len(acceptance_findings([])), 0)
        self.assertGreater(len(acceptance_findings(valid()[:1])), 0)

    def test_misleading_lookalike_host_fails(self):
        reports = valid()
        reports[0]["final"][-1]["host"] = "www.ipcc.ch.attacker.example"
        self.assertTrue(any("IPCC" in item for item
                            in acceptance_findings(reports)))


if __name__ == "__main__":
    unittest.main()
