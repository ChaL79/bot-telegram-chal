"""Deterministic, conservative quality checks for /deepdive.

No additional model calls or network access. Results describe evidence coverage,
not a guarantee that the model's conclusions are true.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit


# Explicit institutional/source names are only a conservative subset. Do not
# infer that a question lacking one of these names has complete source coverage.
SOURCE_REGISTRY = {
    "NOAA": (r"\bNOAA\b", ("noaa.gov", "climate.gov")),
    "NASA": (r"\bNASA\b", ("nasa.gov",)),
    "IPCC": (r"\bIPCC\b", ("ipcc.ch",)),
    "W3C": (r"\bW3C\b", ("w3.org",)),
    "MDN": (r"\bMDN\b", ("developer.mozilla.org",)),
    "GOV.UK": (r"\bGOV[.\s-]*UK\b", ("gov.uk",)),
    "Ollama": (r"\bOllama\b", ("ollama.com",)),
    "Intel": (r"\bIntel\b", ("intel.com",)),
    "TP-Link": (r"\bTP[\s-]*Link\b", ("tp-link.com",)),
    "Wi-Fi Alliance": (r"\bWi[\s-]*Fi\s+Alliance\b", ("wi-fi.org",)),
    "NIST": (r"\bNIST\b", ("nist.gov",)),
    "CISA": (r"\bCISA\b", ("cisa.gov",)),
    "CDC": (r"\bCDC\b", ("cdc.gov",)),
    "HSE": (r"\bHSE\b", ("hse.gov.uk",)),
}


def requested_organizations(question: str) -> dict[str, tuple[str, ...]]:
    """Only return an organization when explicitly named in the question."""
    return {name: domains for name, (pattern, domains) in SOURCE_REGISTRY.items()
            if re.search(pattern, question, re.IGNORECASE)}


def source_on_domains(source: dict, domains: tuple[str, ...]) -> bool:
    """Hostname matches exact organization domain or one of its subdomains."""
    try:
        hostname = (urlsplit(source.get("url", "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return any(hostname == domain or hostname.endswith("." + domain)
               for domain in domains)


def is_verified(source: dict) -> bool:
    return (source.get("read_status") == "verified_page_excerpt"
            and bool(str(source.get("excerpt") or "").strip()))


def assess_sources(question: str, sources: list[dict]) -> dict:
    required = requested_organizations(question)
    verified = [source for source in sources if is_verified(source)]
    covered = [name for name, domains in required.items()
               if any(source_on_domains(source, domains) for source in verified)]
    missing = [name for name in required if name not in covered]
    return {
        "verified_count": len(verified),
        "required": list(required),
        "covered": covered,
        "missing": missing,
        "status": ("INSUFFICIENT_EVIDENCE" if not verified else
                   "PARTIAL" if missing else "PASS"),
    }


# Only check figures with numerical units or currencies, avoiding all plain
# years, reference numbers, numbered lists and arbitrary numbers in prose.
_QUANTITY = re.compile(
    r"(?<![\w])(?:US\$|BND\$|RM\s*|USD\s*|BND\s*|\$\s*)?"
    r"\d+(?:[.,]\d+)?(?:\s*[–-]\s*\d+(?:[.,]\d+)?)?"
    r"\s*(?:%|mm\b|cm\b|km\b|met(?:er|re)s?\b|GB\b|TB\b|Mbps\b|"
    r"billion\b|million\b|juta\b|bilion\b|[A-Z]{0}USD\b)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numeric_values(s: str) -> set[str]:
    # Representation-normalization is intentionally limited. A mismatch is a
    # review warning, never evidence that a figure is necessarily false.
    return {x.replace(",", ".") for x in _NUMBER.findall(s)}


def assess_answer(question: str, answer: str, sources: list[dict]) -> dict:
    report = assess_sources(question, sources)
    citations = {int(x) for x in re.findall(r"\[(\d{1,2})\]", answer)}
    unsupported_cites = sorted(
        n for n in citations if n < 1 or n > len(sources) or not is_verified(sources[n - 1])
    )
    evidence = " ".join(str(source.get("excerpt") or "")
                        for source in sources if is_verified(source))
    known_numbers = _numeric_values(evidence + " " + question)
    unsupported_quantities = []
    for match in _QUANTITY.finditer(answer):
        quantity = match.group().strip()
        if not _numeric_values(quantity).issubset(known_numbers):
            if quantity not in unsupported_quantities:
                unsupported_quantities.append(quantity)
    report["unsupported_citations"] = unsupported_cites
    report["unmatched_quantities"] = unsupported_quantities[:5]
    if report["status"] == "PASS" and (unsupported_cites or unsupported_quantities):
        report["status"] = "PARTIAL"
    return report


def quality_header(report: dict) -> str:
    """Short, truthful qualifier. Never claims full factual verification."""
    status = report["status"]
    if status == "INSUFFICIENT_EVIDENCE":
        return ("Research Quality Gate: INSUFFICIENT_EVIDENCE — "
                "tiada petikan halaman relevan yang berjaya dibaca.")
    description = [f"petikan sah: {report['verified_count']}"]
    if report["missing"]:
        description.append("organisasi belum disahkan: " + ", ".join(report["missing"]))
    if report.get("unsupported_citations"):
        ids = ", ".join(f"[{n}]" for n in report["unsupported_citations"])
        description.append("rujukan tanpa petikan sah: " + ids)
    if report.get("unmatched_quantities"):
        description.append(
            "angka perlu semakan: " + ", ".join(report["unmatched_quantities"][:3]))
    return (f"Research Quality Gate: {status} — " + "; ".join(description)
            + ". Status ini menilai bukti yang tersedia, bukan menjamin semua dakwaan benar.")
