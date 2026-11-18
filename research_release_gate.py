"""Read-only release acceptance policy, independent of workflow green status.

Strict acceptance is not a code unit test: public websites can be unavailable.
A failed live evidence gate blocks merge/deployment; it must NEVER be relaxed
to make CI green. Only owner approval can authorize release after the gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def acceptance_findings(reports: list[dict]) -> list[str]:
    """Permit only two independently corroborated, explicitly scoped pathways.

    An IPCC *data projection* via NASA does NOT establish full IPCC AR6 text.
    ASEAN founding history from its original signatory governments is
    authoritative historical evidence, NOT an ASEAN Secretariat page.
    Keep source coverage as PARTIAL; only scoped release eligibility can PASS.
    """
    if not isinstance(reports, list):
        return ["Invalid report format"]
    by_case = {report.get("case"): report for report in reports
               if isinstance(report, dict)}
    expected = {"NOAA_NASA_IPCC", "OLLAMA_LOCAL_CLOUD", "GENERAL_HISTORY"}
    findings = [f"Missing case: {name}" for name in sorted(expected - set(by_case))]

    def from_domain(host: str, domain: str) -> bool:
        return host == domain or host.endswith("." + domain)

    def verified(report: dict) -> list[dict]:
        return [r for r in report.get("final", []) if isinstance(r, dict)
                and r.get("status") == "verified_page_excerpt"]

    climate = by_case.get("NOAA_NASA_IPCC", {})
    cv = verified(climate)
    noaa = any(from_domain(r.get("host", ""), "noaa.gov")
               or from_domain(r.get("host", ""), "climate.gov") for r in cv)
    nasa = any(from_domain(r.get("host", ""), "nasa.gov") for r in cv)
    original_ipcc = any(from_domain(r.get("host", ""), "ipcc.ch") for r in cv)
    scoped_ipcc = any(
        r.get("host") == "sealevel.nasa.gov"
        and r.get("tier") == "delegated_official"
        and r.get("scope") == "IPCC_AR6_NASA_DATA_ONLY"
        and r.get("supports_ipcc_ar6_sea_level") is True
        for r in cv
    )
    if not noaa or not nasa:
        findings.append("Climate: official NOAA and NASA evidence required")
    if not original_ipcc and not scoped_ipcc:
        findings.append("IPCC: neither original primary excerpt nor scoped NASA AR6 projection verified")
    if climate.get("status") != ("PASS" if original_ipcc else "PARTIAL"):
        findings.append("Climate: provenance/coverage status inconsistent")
    if not original_ipcc and climate.get("missing_requested") != ["IPCC"]:
        findings.append("Climate: inaccessible original IPCC source must be declared")

    ollama = by_case.get("OLLAMA_LOCAL_CLOUD", {})
    ov = verified(ollama)
    hosts = {r.get("host", "") for r in ov
             if from_domain(r.get("host", ""), "ollama.com")}
    if ollama.get("status") != "PASS" or len(ov) < 2 or not (
        any(from_domain(host, "docs.ollama.com") for host in hosts)
        and any(host == "ollama.com" for host in hosts)
    ):
        findings.append("Ollama: independent official docs and pricing required")

    history = by_case.get("GENERAL_HISTORY", {})
    hv = verified(history)
    secretariat = any(from_domain(r.get("host", ""), "asean.org")
                      for r in hv)
    thai = any(
        r.get("host") == "asean.mfa.go.th"
        and r.get("tier") == "founding_government"
        and r.get("supports_asean_founding_1967") is True for r in hv
    )
    singapore = any(
        r.get("host") == "www.mfa.gov.sg"
        and r.get("tier") == "founding_government"
        and r.get("supports_asean_founding_1967") is True for r in hv
    )
    if not secretariat and not (thai and singapore):
        findings.append("ASEAN: original Secretariat or two verified founding governments required")
    if history.get("status") != ("PASS" if secretariat else "PARTIAL"):
        findings.append("ASEAN: provenance/coverage status inconsistent")
    if not secretariat and history.get("missing_requested") != ["ASEAN"]:
        findings.append("ASEAN: original Secretariat absence must be declared")
    return findings

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    reports = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(reports, list):
        raise SystemExit("Invalid acceptance report format")
    findings = acceptance_findings(reports)
    if findings:
        print("LIVE_RELEASE_GATE: BLOCKED")
        for finding in findings:
            print("RELEASE_BLOCKER:", finding)
        raise SystemExit(1)
    print("LIVE_RELEASE_GATE: SCOPED EVIDENCE PASS; original-source gaps remain PARTIAL; OWNER APPROVAL STILL REQUIRED")


if __name__ == "__main__":
    main()
