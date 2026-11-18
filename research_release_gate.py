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
    by_case = {report.get("case"): report for report in reports}
    missing = {"NOAA_NASA_IPCC", "OLLAMA_LOCAL_CLOUD", "GENERAL_HISTORY"} - set(by_case)
    findings = [f"Missing case: {name}" for name in sorted(missing)]

    climate = by_case.get("NOAA_NASA_IPCC", {})
    agencies = {"noaa.gov", "climate.gov", "nasa.gov", "ipcc.ch"}
    hosts = {ref.get("host", "").lower() for ref in climate.get("final", [])
             if ref.get("status") == "verified_page_excerpt"}
    def from_domain(host: str, domain: str) -> bool:
        return host == domain or host.endswith("." + domain)
    for org, domains in (("NOAA", {"noaa.gov", "climate.gov"}),
                         ("NASA", {"nasa.gov"}),
                         ("IPCC", {"ipcc.ch"})):
        if not any(from_domain(host, domain) for host in hosts for domain in domains):
            findings.append(f"{org}: no directly verified official agency HTML")
    if climate.get("status") != "PASS":
        findings.append("Climate comparison is not PASS")

    ollama = by_case.get("OLLAMA_LOCAL_CLOUD", {})
    ollama_hosts = {ref.get("host", "") for ref in ollama.get("final", [])
                    if ref.get("status") == "verified_page_excerpt"}
    if ollama.get("status") != "PASS" or not any(
        from_domain(host, "ollama.com") for host in ollama_hosts
    ) or (ollama.get("final_verified") or 0) < 2:
        findings.append("Ollama: fewer than two verified sources or not PASS")

    history = by_case.get("GENERAL_HISTORY", {})
    if not any(from_domain(ref.get("host", ""), "asean.org")
               for ref in history.get("final", [])
               if ref.get("status") == "verified_page_excerpt"):
        findings.append("ASEAN: no verified official ASEAN history HTML")
    if history.get("status") != "PASS":
        findings.append("ASEAN history status is not PASS")
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
    print("LIVE_RELEASE_GATE: TECHNICAL EVIDENCE PASS; OWNER APPROVAL STILL REQUIRED")


if __name__ == "__main__":
    main()
