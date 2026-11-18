"""Small reviewed public-source catalog used only if general discovery is weak.

These are independently checked FIRST-PARTY or government-run HTML pages.
They are NOT evidence until read through the existing DNS-pinned HTTPS
reader and accepted as topically relevant. No query-string/redirect bypass,
PDF download, proxy or paid service is permitted.
"""
from __future__ import annotations

import re
from public_article_reader import safe_public_article_url

# The hosted IPCC Data Distribution Centre is an IPCC data repository, NOT a
# substitute for the full AR6 Summary for Policymakers text on www.ipcc.ch.
IPCC_DDC_SEA_LEVEL = (
    "https://ipcc-data.climate.columbia.edu/content/ipcc-ar6-syr-lr-figure-34"
)
IPCC_AR6_NASA_TOOL = "https://sealevel.nasa.gov/data_tools/17/"

ASEAN_HISTORY = (
    ("https://asean.org/the-founding-of-asean/",
     "ASEAN Secretariat: The Founding of ASEAN",
     "official"),
    ("https://asean.mfa.go.th/en/page/62206-the-birth-of-asean",
     "Thai MFA ASEAN office: Birth of ASEAN",
     "government"),
    ("https://www.mfa.gov.sg/about-mfa/foreign-policy/international-organisations/association-of-southeast-asian-nations-asean/",
     "Singapore MFA: Association of Southeast Asian Nations (ASEAN)",
     "government"),
)
ASEAN_GENERAL = (
    ("https://asean.org/about-asean/",
     "ASEAN Secretariat: About ASEAN",
     "official"),
)


def reviewed_candidates(question: str, uncovered: list[str]) -> list[dict]:
    """Select scoped reviewed URLs, not arbitrary user-supplied URLs."""
    if not isinstance(question, str) or len(question) > 3500:
        return []
    q = question.casefold()
    selected = []
    if "IPCC" in uncovered and re.search(r"\b(?:sea\s+level|paras\s+laut)\b", q):
        selected.append({
            "provider": "IPCC Data Distribution Centre (Columbia-hosted; not full IPCC report)",
            "title": "IPCC AR6 sea-level observations and projections (DDC Figure 3.4)",
            "url": IPCC_DDC_SEA_LEVEL,
            "snippet": "",
            "published": "",
            "recovery_for": ["IPCC"],
            "source_tier": "delegated_official",
            "source_scope": "IPCC_DDC_DATA_ONLY",
        })
    if "IPCC" in uncovered and re.search(r"\\b(?:sea\\s+level|paras\\s+laut)\\b", q):
        selected.append({
            "provider": "NASA Sea Level IPCC AR6 projection data (delegated; not IPCC report)",
            "title": "NASA: IPCC AR6 Sea Level Projection Tool",
            "url": IPCC_AR6_NASA_TOOL,
            "snippet": "", "published": "",
            "recovery_for": ["IPCC"],
            "source_tier": "delegated_official",
            "source_scope": "IPCC_AR6_NASA_DATA_ONLY",
        })
    if "ASEAN" in uncovered:
        if re.search(r"\b(?:sejarah|history|founding|foundation|establish(?:ment|ed)?|"
                     r"pembentukan|penubuhan|penubuhannya|origin|bangkok)\b", q):
            chosen = ASEAN_HISTORY
        else:
            chosen = ASEAN_GENERAL
        for url, title, tier in chosen:
            selected.append({
                "provider": ("ASEAN Secretariat official reference"
                             if tier == "official" else
                             "Singapore Government ASEAN background reference"),
                "title": title,
                "url": url,
                "snippet": "",
                "published": "",
                "recovery_for": ["ASEAN"] if tier == "official" else [],
                "source_tier": tier,
                "source_scope": "ASEAN_BACKGROUND",
            })
    return [ref for ref in selected
            if safe_public_article_url(ref["url"])]
