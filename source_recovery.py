"""Bounded targeted recovery for /deepdive after source coverage checks.

Uses public-search-discovered HTTPS HTML only, with the existing DNS-pinned
reader. It neither bypasses denied hosts nor treats search snippets as evidence.
No additional Gemini calls, paid APIs, raw user URLs, or PDF downloads.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import urlsplit

import httpx

from free_research import public_source_search, search_query_terms
from public_article_reader import read_public_html, safe_public_article_url
from research_quality import assess_sources, requested_organizations, source_on_domains
from official_evidence_catalog import reviewed_candidates
from source_reader import MainText, BoundedBodyText, select_relevant_excerpt

log = logging.getLogger("source_recovery")

MAX_RECOVERY_READS = 2
MAX_RECOVERY_RESULTS = 8
RECOVERY_DEADLINE_SECONDS = 9.0
TARGETED_SEARCH_SECONDS = 5.0
SOURCE_READER_FLAG = "ENABLE_GENERAL_PUBLIC_READER"

_GENERIC_TERMS = {
    "research", "study", "studies", "compare", "comparison", "source", "sources",
    "evidence", "proof", "review", "official", "data", "latest", "update",
    "about", "explain", "analysis", "between", "versus", "find", "check",
    "bandingkan", "bukti", "punca", "utama", "tentang", "antara", "semak",
}


def _topic_terms(question: str) -> set[str]:
    org_tokens = set()
    for name in requested_organizations(question):
        org_tokens.update(search_query_terms(name))
    return {term for term in search_query_terms(question)
            if len(term) >= 3 and term not in org_tokens
            and term not in _GENERIC_TERMS}


def _discovered_candidates(
    question: str,
    discovered: list[dict],
    prior: list[dict],
    missing: list[str],
) -> list[dict]:
    """Select existing search-index URLs; never convert question text to URLs."""
    required = requested_organizations(question)
    topic = _topic_terms(question)
    attempted_urls = {s.get("url") for s in prior
                      if s.get("read_status") in ("page_unavailable", "verified_page_excerpt")}
    denied_hosts = {urlsplit(s["url"]).hostname
                    for s in prior if s.get("read_status") == "page_unavailable"
                    and s.get("url") and safe_public_article_url(s["url"])}
    candidates = []
    seen = set()
    for ref in discovered:
        url = ref.get("url", "")
        if (url in attempted_urls or url in seen or
                not safe_public_article_url(url)):
            continue
        host = urlsplit(url).hostname
        # Prevent lookalike institutional subdomains (e.g.
        # ipcc.ch.attacker.example) from being presented even as secondary
        # research evidence. A genuine official subdomain ends in .ipcc.ch.
        if any(domain + "." in (host or "") and
               not ((host or "") == domain or
                    (host or "").endswith("." + domain))
               for domains in requested_organizations(question).values()
               for domain in domains):
            continue
        if host in denied_hosts:
            # An unavailable host is not retried on another URL or via a proxy.
            continue
        metadata = " ".join((str(ref.get("title", "")),
                             str(ref.get("snippet", "")),
                             urlsplit(url).path.replace("-", " ")))
        metadata_terms = set(search_query_terms(metadata))
        overlap = len(topic & metadata_terms)
        if topic and overlap < min(2, len(topic)):
            continue
        matched_official = [name for name in missing
                            if source_on_domains(ref, required[name])]
        matched_secondary = [
            name for name in missing
            if re.search(r"\b" + re.escape(name).replace(r"\-", r"[-\s]") + r"\b",
                         metadata, re.IGNORECASE)
            and name not in matched_official
        ]
        if missing and not (matched_official or matched_secondary):
            continue
        if missing:
            matched = matched_official or matched_secondary
            tier = "official" if matched_official else "secondary"
            rank = (100 if matched_official else 20) + overlap
        else:
            matched = []
            tier = "other"
            rank = overlap
        seen.add(url)
        candidates.append((rank, {**ref, "recovery_for": matched,
                                  "source_tier": tier}))
    # Stable ordering avoids nondeterminism in citations and tests.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [ref for _, ref in candidates]


def _scoped_query(question: str, missing: list[str]) -> str:
    topic = list(dict.fromkeys(search_query_terms(question)))
    organization_terms = set()
    for name in requested_organizations(question):
        organization_terms.update(search_query_terms(name))
    subject = [t for t in topic if t not in organization_terms and
               t not in _GENERIC_TERMS and len(t) >= 3]
    # For a general question with no named institution, use a short topical
    # search. This makes one bounded recovery attempt if all indexes were empty.
    prefix = missing[0] if missing else ""
    return (prefix + " " + " ".join(subject[:5])).strip()[:150]


async def _read_recovery_ref(
    question: str, ref: dict, client: httpx.AsyncClient | None
) -> dict:
    """One safe attempt. A failed read must never masquerade as metadata."""
    failed = {**ref, "read_status": "page_unavailable"}
    url = ref["url"]
    try:
        html = await read_public_html(url, client=client)
        if not html:
            return failed
        parser = MainText()
        parser.feed(html)
        body = parser.content()
        if not body:
            fallback = BoundedBodyText()
            fallback.feed(html)
            body = fallback.content()
        # Reviewed multilingual history questions can use English-language
        # grounding terms, while the actual page must still contain them.
        focus = ("ASEAN founding regional cooperation 1967 history"
                 if ref.get("source_scope") == "ASEAN_BACKGROUND" else
                 "IPCC AR6 sea level rise projections"
                 if ref.get("source_scope") == "IPCC_DDC_DATA_ONLY" else
                 question)
        excerpt = select_relevant_excerpt(body, focus)
        if not excerpt or len(excerpt) < 65:
            return failed
        if ref.get("source_scope") == "ASEAN_BACKGROUND":
            terms = set(re.findall(r"[a-z0-9]+", excerpt.casefold()))
            if "asean" not in terms or not (terms & {
                    "founding", "founded", "established", "bangkok", "1967"}):
                return failed
        elif ref.get("source_scope") == "IPCC_DDC_DATA_ONLY":
            terms = set(re.findall(r"[a-z0-9]+", excerpt.casefold()))
            if not ({"ipcc", "sea", "level"} <= terms):
                return failed
        else:
            topic = _topic_terms(question)
            actual_terms = set(search_query_terms(excerpt))
            if topic and len(topic & actual_terms) < min(2, len(topic)):
                return failed
        if ref.get("source_tier") == "secondary":
            # A third-party article must explicitly mention the organization
            # in ACTUAL page text, not merely its search-result metadata.
            if not all(re.search(r"\b" + re.escape(name) + r"\b", excerpt,
                                 re.IGNORECASE)
                       for name in ref.get("recovery_for", [])):
                return failed
        label = ("Targeted recovery: verified IPCC DDC dataset description (not AR6 report)"
                 if ref.get("source_tier") == "delegated_official" else
                 "Targeted recovery: verified first-party article"
                 if ref.get("source_tier") == "official" else
                 "Targeted recovery: verified secondary article (not primary)"
                 if ref.get("source_tier") == "secondary" else
                 "Targeted recovery: verified public article")
        log.info("Recovery excerpt validated host=%s tier=%s",
                 urlsplit(url).hostname, ref.get("source_tier"))
        return {**ref, "provider": label,
                "read_status": "verified_page_excerpt", "excerpt": excerpt}
    except (ValueError, UnicodeError):
        return failed


def _merge_results(initial: list[dict], recovered: list[dict]) -> list[dict]:
    """Promote newly read excerpts while keeping original failed evidence."""
    verified_original = [s for s in initial if s.get("read_status") == "verified_page_excerpt"]
    verified_new = [s for s in recovered if s.get("read_status") == "verified_page_excerpt"]
    rest = [s for s in initial if s.get("read_status") != "verified_page_excerpt"]
    failures_new = [s for s in recovered if s.get("read_status") != "verified_page_excerpt"]
    output = []
    seen = set()
    for ref in verified_original + verified_new + rest + failures_new:
        url = ref.get("url", "")
        key = url or ("missing", ref.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(ref)
        if len(output) >= MAX_RECOVERY_RESULTS:
            break
    return output


async def recover_sources(
    question: str,
    discovered: list[dict],
    initial: list[dict],
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Try at most two additional pages and one bounded public discovery pass.

    Recovery is driven by missing named organizations, or zero verified excerpts
    with additional unvisited relevant search results. Never retries a
    previously blocked host. No non-HTML reads or browser access.
    """
    if os.environ.get(SOURCE_READER_FLAG, "1") != "1":
        return initial

    status = assess_sources(question, initial)
    missing = status["missing"]
    if not missing and status["verified_count"]:
        return initial
    candidates = _discovered_candidates(question, discovered, initial, missing)
    # Reviewed institutional pages enter only through the same DNS-pinned HTML
    # reader as public-search results. A catalog entry is never verified by fiat.
    attempted = {s.get("url") for s in initial}
    known_bad_hosts = {urlsplit(s["url"]).hostname
                       for s in initial
                       if s.get("read_status") == "page_unavailable"
                       and safe_public_article_url(s.get("url", ""))}
    reviewed = [
        entry for entry in reviewed_candidates(question, missing)
        if entry["url"] not in attempted
        and urlsplit(entry["url"]).hostname not in known_bad_hosts
    ]
    # Prefer scoped primary references; delegated IPCC datasets are not the
    # same as the primary assessment report and cannot fill official coverage.
    candidates = reviewed + [
        candidate for candidate in candidates
        if candidate["url"] not in {ref["url"] for ref in reviewed}
    ]
    # One targeted search when the named official organization remains
    # uncovered, OR a general research question had zero viable candidates.
    # No broad discovery loops, no paid APIs and no repeat of failed hosts.
    needs_search = (
        (bool(missing) and not any(
            x.get("source_tier") == "official" for x in candidates))
        or (not missing and not candidates)
    )
    if reviewed:
        needs_search = False
    if needs_search and _scoped_query(question, missing):
        try:
            extra = await asyncio.wait_for(
                public_source_search(_scoped_query(question, missing),
                                     client=client),
                timeout=TARGETED_SEARCH_SECONDS,
            )
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError):
            extra = []
        discovered_again = _discovered_candidates(
            question, discovered + extra, initial, missing
        )
        candidates = reviewed + [
            candidate for candidate in discovered_again
            if candidate["url"] not in {ref["url"] for ref in reviewed}
        ]

    attempts = candidates[:MAX_RECOVERY_READS]
    if not attempts:
        return initial

    # Reader's DNS-pinned HTTPS transport is unchanged. The deadline also
    # leaves headroom for the single existing Gemini synthesis request.
    try:
        recovered = await asyncio.wait_for(
            asyncio.gather(*(_read_recovery_ref(question, ref, client)
                             for ref in attempts)),
            timeout=RECOVERY_DEADLINE_SECONDS,
        )
    except asyncio.TimeoutError:
        log.info("Source recovery deadline exceeded")
        return initial
    return _merge_results(initial, recovered)
