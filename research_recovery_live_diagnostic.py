"""Live, read-only research recovery diagnostic (no Gemini or secrets).

This is deliberately a diagnostic, not a claim that third-party sites will
always allow indexing/fetching from GitHub-hosted runners. Real-world coverage
is reported separately from program invariants.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from free_research import public_source_search
from source_reader import enrich_with_read_excerpts
from source_recovery import recover_sources
from research_quality import assess_sources, is_verified

# Include unrelated domains to catch overfitting to climate or product pricing.
CASES = (
    ("NOAA_NASA_IPCC",
     "Bandingkan bukti NOAA NASA dan IPCC tentang kenaikan paras laut, "
     "punca utama, unjuran dan ketidakpastian."),
    ("OLLAMA_LOCAL_CLOUD",
     "Bandingkan Ollama local AI dan Ollama cloud AI dari segi harga, "
     "perkakasan, privasi, prestasi dan akaun."),
    ("GENERAL_HISTORY",
     "Jelaskan sejarah pembentukan ASEAN dan tujuan penubuhannya "
     "berdasarkan sumber sejarah yang boleh dibaca."),
)


async def inspect_case(name: str, question: str) -> dict:
    search_hits = await public_source_search(question)
    initial = await enrich_with_read_excerpts(question, search_hits)
    final = await recover_sources(question, search_hits, initial)
    assert len(final) <= 8, "Unbounded evidence list"
    assert all(isinstance(s.get("url", ""), str) for s in final)
    for source in final:
        if is_verified(source):
            assert source.get("excerpt") and len(source["excerpt"].strip()) >= 30
            assert source.get("url", "").startswith("https://")
        else:
            assert source.get("read_status") != "verified_page_excerpt", (
                "Excerpt-free source incorrectly labeled verified")
    quality = assess_sources(question, final)
    return {
        "case": name,
        "search_hits": len(search_hits),
        "initial_verified": sum(is_verified(s) for s in initial),
        "final_verified": quality["verified_count"],
        "status": quality["status"],
        "missing_requested": quality["missing"],
        "final": [
            {"host": urlsplit(s.get("url", "")).hostname or "",
             "status": s.get("read_status", "search_metadata_only"),
             "tier": s.get("source_tier", "initial"),
             "recovered_for": s.get("recovery_for", []),
             "title": str(s.get("title", ""))[:80]}
            for s in final
        ],
    }


async def main() -> None:
    results = []
    for name, question in CASES:
        try:
            report = await asyncio.wait_for(inspect_case(name, question), timeout=45)
        except asyncio.TimeoutError:
            report = {"case": name, "status": "INCONCLUSIVE_TIMEOUT"}
        results.append(report)
        print("LIVE_CASE " + json.dumps(report, ensure_ascii=False), flush=True)
    # Report counts, not a blanket assertion that all external sites are alive.
    counts = {name: sum(x.get("status") == name for x in results)
              for name in ("PASS", "PARTIAL", "INSUFFICIENT_EVIDENCE",
                           "INCONCLUSIVE_TIMEOUT")}
    print("LIVE_SUMMARY " + json.dumps(counts), flush=True)
    Path("research_acceptance.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
