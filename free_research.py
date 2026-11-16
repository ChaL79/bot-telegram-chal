"""Low-volume, no-key public source discovery.

RSS and MediaWiki providers may be unavailable or incomplete. This supplies
search-result metadata, NOT verification of the article/page contents.
No arbitrary user-supplied URLs are fetched, so search results cannot SSRF the bot.
"""
import asyncio
import html
import ipaddress
import json
import logging
import re
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree

import httpx
from public_article_reader import safe_public_article_url

log = logging.getLogger("free_research")
MAX_BYTES = 550_000
MAX_SOURCES = 8

# No open-web crawling: discovery can focus on real official institution sites,
# and the reader only fetches article URLs on these exact, trusted hosts.
# This is an extensible institution-domain registry, not a guessed URL list.
OFFICIAL_SCIENCE_SITES = {
    "noaa": ("www.climate.gov", "www.noaa.gov"),
    "nasa": ("sealevel.nasa.gov", "science.nasa.gov"),
    "ipcc": ("www.ipcc.ch", "ipcc.ch"),
}
OFFICIAL_SCIENCE_HOSTS = frozenset(
    host for hosts in OFFICIAL_SCIENCE_SITES.values() for host in hosts
)
SCIENCE_TOPIC_WORDS = frozenset({
    "sea", "level", "rise", "rising", "thermal", "expansion",
    "ocean", "warming", "glaciers", "glacier", "ice", "sheets", "melting",
    "climate", "temperature", "water",
})

# Organization-level search targeting (not fixed article URLs). These are
# only READ-ONLY queries sent to the public RSS search provider.
# A separate DNS-pinned reader must validate any retrieved article URL.
ACCESSIBILITY_SOURCE_HOSTS = {
    "w3c": "www.w3.org",
    "mdn": "developer.mozilla.org",
    "govuk": "www.gov.uk",
}
ACCESSIBILITY_TOPIC_WORDS = frozenset({
    "wcag", "accessibility", "accessible", "keyboard", "focus",
    "visibility", "visible", "obscured", "testing", "test",
})
WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChaL79Bot/1.4; personal research)",
    "Accept": "application/rss+xml,application/xml,application/json,text/xml;q=0.8,*/*;q=0.1",
}

# Search tasks use natural-language instructions in Malay and English.
# Query terms must omit command words, otherwise "Cari" dominates Bing results.
QUERY_FILLER = frozenset({
    "cari", "carikan", "sumber", "tentang", "mengenai", "berkaitan", "di",
    "dalam", "dari", "daripada", "untuk", "yang", "dan", "atau", "serta", "kepada",
    "saya", "anda", "awak", "kamu", "tolong", "sila", "buat", "berikan", "beri",
    "ringkaskan", "ringkasan", "dapatan", "bezakan", "fakta", "andaian",
    "sertakan", "pautan", "hasil", "carian", "rujukan", "artikel",
    "dokumentasi", "maklumat", "tunjukkan", "semak", "periksa", "kajian",
    "search", "check", "please", "find", "source", "sources", "about",
    "the", "a", "an", "of", "for", "with", "and", "or", "summarize", "compare",
    "deep", "dive", "research", "explain", "how", "what",
    "verify", "verified", "verification", "official", "whether", "from",
    "identify", "include", "documented", "documentation", "assumptions",
    "assumption", "actual", "read", "reading", "pages", "page", "provide",
    "specifically", "specific", "exactly", "exact", "particular", "separately",
    "conduct", "independent", "multi", "source", "research", "review", "studies",
    "publications", "publication", "methods", "method", "three", "account",
    "as", "at", "of", "my", "its", "vs", "is", "on", "for", "which",
    "november", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "december",
    "each", "across", "where", "include", "clearly", "explain",
    "without", "paid", "product", "products", "evidence", "actual",
    "using", "answer", "questions", "organization", "organizations",
    "website", "websites", "web", "docs", "guide", "guides",
    # Malay prompt scaffolding must not displace technical subject terms.
    "bandingkan", "berdasarkan", "rasmi", "dengan", "perbezaan",
    "sebenar", "cuba", "baca", "kandungan", "sekurang", "kurangnya",
    "dua", "tiga", "berbeza", "setiap", "halaman", "berjaya",
    "asingkan", "metadata", "sahaja", "gagal", "diakses", "jangan",
    "reka", "keputusan", "ujian", "petikan",
})


def _terms(value: str) -> list[str]:
    # Keep hyphenated product names and their version as one search concept.
    # "Wi-Fi 7", "WiFi7", and "WiFi 7" all get wifi7 + wifi; the same
    # normalization is applied to titles/snippets when scoring RSS hits.
    value = value.casefold()
    value = re.sub(r"\bwi[\s-]*fi\b", "wifi", value)
    value = re.sub(r"\btp[\s-]*link\b", "tplink", value)
    value = re.sub(r"\bmulti[\s-]*link\b", "multilink", value)
    value = re.sub(r"\b([256])\s*ghz\b", r"\1ghz", value)
    value = re.sub(r"\bwifi[\s-]*([67])\b", r"wifi\1 wifi", value)
    return [word for word in re.findall(r"[a-z0-9]+", value)
            if word not in QUERY_FILLER and len(word) > 1]


# Prefer named topical concepts, even if the user includes a long introductory
# sentence or multi-step instruction list before naming the real topic. This
# avoids accidental searches for words like "conduct", "independent", or dates.
TOPIC_ANCHORS = frozenset({
    "aiolimiter", "asyncio", "semaphore", "python", "javascript", "kubernetes",
    "gemini", "aistudio", "antigravity", "notebooklm", "groq", "openrouter",
    "passkeys", "passkey", "fido2", "totp", "mfa", "otp", "phishing",
    "authenticator", "authentication", "webauthn", "cybersecurity",
    "nist", "cisa", "fido", "sms", "sim", "swapping",
    "noaa", "nasa", "ipcc", "sea", "level", "rise",
    "thermal", "expansion", "ocean", "glaciers", "glacier",
    "ice", "sheets", "melting", "climate",
    "wcag", "accessibility", "keyboard", "focus", "visibility",
    "obscured", "w3c", "mdn", "govuk",
    # Cross-domain consumer/technical research; not a vendor URL allowlist.
    "wifi7", "wifi6", "wifi", "mlo", "6ghz", "latency", "multilink",
})
IDENTITY_ANCHORS = frozenset({
    "passkeys", "passkey", "fido2", "totp", "mfa", "otp", "phishing",
    "authenticator", "webauthn", "sms",
})


def search_query_terms(request: str) -> list[str]:
    """Extract relevant terms across a complete question; no LLM, no extra API."""
    request = clean(request, 2800)
    request = re.sub(r"^/(?:deepdive|power)\b\s*", "", request, flags=re.I)
    words = [x for x in _terms(request) if not x.isdecimal()]
    anchors = []
    for word in words:
        if word in TOPIC_ANCHORS and word not in anchors:
            anchors.append(word)
    # When researching authentication, generic words like "api", "bot",
    # "telegram", "free" are context only, not the topic of a web search.
    if IDENTITY_ANCHORS.intersection(anchors):
        selected = [x for x in anchors if x in IDENTITY_ANCHORS]
        selected += [x for x in anchors if x in {"nist", "cisa", "fido", "authentication", "cybersecurity"}]
    else:
        selected = anchors[:]
    for word in words:
        if word not in selected and word not in QUERY_FILLER:
            selected.append(word)
        if len(selected) >= 12:
            break
    return selected[:12]


TECH_KEYWORDS = frozenset({
    "python", "asyncio", "javascript", "typescript", "java", "rust", "golang",
    "github", "docker", "kubernetes", "api", "programming", "coding",
    "security", "cybersecurity", "linux", "regex", "sql", "algorithm",
    "rate", "limiting", "limiter", "limit", "semaphore", "token", "telegram", "bot", "llm",
    "aiolimiter", "async", "leaky", "bucket", "sliding", "window", "burst", "throttle", "throttling",
    "passkeys", "passkey", "fido2", "totp", "mfa", "otp", "phishing",
    "authenticator", "authentication", "webauthn", "cybersecurity",
})


def _canonical(term: str) -> str:
    """Minimal normalization to match rate limit vs rate limiting."""
    if term in ("limit", "limits", "limited", "limiting", "limiter"):
        return "limit"
    if term in ("requests", "requesting", "request"):
        return "request"
    if term in ("thread", "threads", "threading"):
        return "thread"
    return term


def technical_topic(terms: list[str]) -> bool:
    return any(word in TECH_KEYWORDS for word in terms)


def parse_stackexchange(payload: dict, limit: int = 6) -> list[dict]:
    """Only Stack Overflow's public question metadata, not accepted answers."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    found = []
    for q in items[:limit * 3]:
        if not isinstance(q, dict):
            continue
        uri = q.get("link", "")
        title = clean(q.get("title", ""), 115)
        if not allowed_source_url(uri) or not title:
            continue
        tags = q.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tag_text = " ".join(clean(t, 25) for t in tags[:7] if isinstance(t, str))
        # Tags are actual metadata; do not claim answers are verified.
        desc = f"Stack Overflow community question. Tags: {tag_text}"
        found.append({
            "provider": "Stack Overflow (community Q&A)",
            "title": title, "url": uri, "snippet": desc[:300],
            "published": "",
        })
        if len(found) >= limit:
            break
    return found


def parse_hackernews(payload: dict, limit: int = 5) -> list[dict]:
    """Headline metadata from Hacker News Algolia; not a primary source."""
    if not isinstance(payload, dict):
        return []
    hits = payload.get("hits")
    if not isinstance(hits, list):
        return []
    found = []
    for hit in hits[:limit * 3]:
        if not isinstance(hit, dict):
            continue
        title = clean(hit.get("title", ""), 115)
        uri = hit.get("url")
        if not allowed_source_url(uri) or not title:
            continue
        found.append({
            "provider": "Hacker News (community link)",
            "title": title, "url": uri,
            "snippet": "Hacker News listing; title only, article not independently checked.",
            "published": clean(hit.get("created_at", ""), 45),
        })
        if len(found) >= limit:
            break
    return found

def source_relevance(item: dict, query_terms: list[str]) -> int:
    """Require core subject matches; a generic 'CARI' result scores zero."""
    if not query_terms:
        return 0
    title_terms = {_canonical(x) for x in _terms(item.get("title", ""))}
    url = urlsplit(item.get("url", ""))
    content_terms = {_canonical(x) for x in _terms(
        item.get("snippet", "") + " " + (url.hostname or "") + " " + url.path
    )}
    sought = [_canonical(x) for x in query_terms]
    matched = set(sought) & (title_terms | content_terms)
    # A Gemini API privacy query must not be satisfied by generic Google
    # search results or unrelated Gemini launches. Require privacy/data-use
    # context as well as a named developer/API subject in the result itself.
    privacy_terms = {"privacy", "confidential", "sensitive", "training", "data",
                     "terms", "policy", "policies"}
    if "gemini" in sought and "api" in sought and privacy_terms & set(sought):
        found = title_terms | content_terms
        if not ({"gemini", "api", "studio"} & found) or not (privacy_terms & found):
            return 0
    # Named libraries/products are hard anchors: generic instructions such as
    # "verify" or "official" must never admit unrelated search results.
    hard_anchors = {"aiolimiter", "asyncio", "semaphore"}
    required = set(sought) & hard_anchors
    # Technical identity questions must mention a real authentication
    # concept rather than just generic login/account words in a snippet.
    if IDENTITY_ANCHORS.intersection(sought):
        named = IDENTITY_ANCHORS | {"authentication", "fido"}
        if not (named & (title_terms | content_terms)):
            return 0
    if required and not (required & matched):
        return 0
    # Requests comparing independent scientific institutions must not accept
    # a result just because it mentions a generic word such as "climate".
    organizations = set(sought) & set(OFFICIAL_SCIENCE_SITES)
    if organizations and len(organizations) >= 2:
        subject = set(sought) & SCIENCE_TOPIC_WORDS
        found = title_terms | content_terms
        if len(found & subject) < min(2, len(subject)):
            return 0
        if url.hostname not in OFFICIAL_SCIENCE_HOSTS:
            return 0
        # A NOAA article naturally says NOAA in its title, but a NASA article
        # should not be forced to include NOAA/IPCC as well.
        return len(matched) + 2 * len(matched & title_terms) + 6
    # Multi-institution accessibility questions should accept an article from
    # EACH named source: the MDN article need not literally say W3C and GOV.UK.
    accessibility_orgs = set(sought) & set(ACCESSIBILITY_SOURCE_HOSTS)
    if {"gov", "uk"}.issubset(set(sought)):
        accessibility_orgs.add("govuk")
    if len(accessibility_orgs) >= 2:
        matching_hosts = {
            ACCESSIBILITY_SOURCE_HOSTS[org] for org in accessibility_orgs
        }
        topical = set(sought) & ACCESSIBILITY_TOPIC_WORDS
        found = title_terms | content_terms
        if url.hostname not in matching_hosts or len(found & topical) < min(2, len(topical)):
            return 0
        return len(matched) + 2 * len(matched & title_terms) + 6
    minimum = 2 if len(set(sought)) >= 3 else len(set(sought))
    if len(matched) < minimum:
        return 0
    # Require subject anchor, rather than broad incidental words.
    if len(sought) >= 3 and not (set(sought[:2]) & matched):
        return 0
    # Title matches are more relevant than incidental snippet/domain tokens.
    return len(matched) + 2 * len(matched & title_terms)


def clean(value: str, size: int = 280) -> str:
    """Plain text, bounded length: feeds and wiki snippets are untrusted input."""
    if not isinstance(value, str):
        return ""
    value = re.sub(r"<[^>]*>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:size]


def allowed_source_url(raw: str) -> bool:
    """Only public HTTPS links in returned citations; never fetch these URLs."""
    if not isinstance(raw, str) or len(raw) > 750 or any(c in raw for c in "\r\n\t<>"):
        return False
    try:
        url = urlsplit(raw)
        host = url.hostname
        if (url.scheme != "https" or not host or url.username or url.password
                or host.lower() in ("localhost", "localhost.localdomain")
                or host.lower().endswith(".local") or "." not in host):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except ValueError:
        return False


def parse_rss(raw: bytes, provider: str, limit: int = 4) -> list[dict]:
    if len(raw) > MAX_BYTES:
        return []
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return []
    items = []
    for item in root.findall("./channel/item")[:limit * 3]:
        uri = (item.findtext("link") or "").strip()
        if not allowed_source_url(uri):
            continue
        title = clean(item.findtext("title") or "", 115)
        desc = clean(item.findtext("description") or "", 300)
        published = clean(item.findtext("pubDate") or "", 55)
        if not title:
            continue
        items.append({"provider": provider, "title": title, "url": uri,
                      "snippet": desc, "published": published})
        if len(items) == limit:
            break
    return items


def parse_wikipedia(data: dict, limit: int = 3) -> list[dict]:
    if not isinstance(data, dict):
        return []
    query = data.get("query", {})
    hits = query.get("search", []) if isinstance(query, dict) else []
    result = []
    for hit in hits[:limit * 2]:
        if not isinstance(hit, dict):
            continue
        title = clean(hit.get("title", ""), 110)
        if not title:
            continue
        url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_-")
        if allowed_source_url(url):
            result.append({"provider": "Wikipedia", "title": title, "url": url,
                           "snippet": clean(hit.get("snippet", ""), 300),
                           "published": ""})
        if len(result) == limit:
            break
    return result


def readable_article_priority(item: dict) -> int:
    """Prefer safe direct article/doc URLs over RSS redirects and marketing landing pages.

    Heuristic only: a nonzero score is NOT evidence that the page can be read.
    The DNS-pinned reader and actual HTML extraction remain authoritative.
    """
    url = item.get("url", "")
    if not safe_public_article_url(url):
        return 0
    path = urlsplit(url).path.casefold()
    segments = [part for part in path.split("/") if part]
    score = 10  # A direct safe URL is preferable to a search index redirect.
    if path.endswith((".html", ".htm")):
        score += 6
    if any(segment in {"resources", "guides", "guide", "article", "articles",
                       "blog", "documentation", "docs"} for segment in segments):
        score += 6
    if len(segments) >= 4:
        score += 3
    elif len(segments) >= 3:
        score += 1
    if urlsplit(url).hostname in {"en.wikipedia.org", "en.m.wikipedia.org"}:
        score -= 4  # Useful metadata, but often large/blocked in bounded reader.
    return score


def prioritize_readable_sources(items: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Source diversity first, without discarding useful secondary metadata."""
    ranked = sorted(items, key=lambda pair: (
        readable_article_priority(pair[1]), pair[0]), reverse=True)
    diverse, others, hosts = [], [], set()
    for pair in ranked:
        host = urlsplit(pair[1]["url"]).hostname
        if host in hosts:
            others.append(pair)
        else:
            hosts.add(host)
            diverse.append(pair)
    return diverse + others


async def public_source_search(query: str, *, client: httpx.AsyncClient | None = None) -> list[dict]:
    """Search Bing RSS, Wikipedia API and Google News RSS concurrently.

    Uses only public, read-only endpoints. Fails gracefully if hosts block Render.
    No private network fetches and no additional paid API keys.
    """
    terms = search_query_terms(query)
    if not terms:
        return []
    query_text = query
    # Search public indexes with core concepts, not a 12-term AND-like
    # sentence containing vendors, commentary, and unrelated instructions.
    query = " ".join(terms[:6])[:160]

    async def search(session: httpx.AsyncClient) -> list[dict]:
        tech = technical_topic(terms)
        # Widen a long query to the core subject for topic-specific public indexes.
        technical_query = " ".join(terms[:4])

        async def get_bing():
            resp = await session.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss", "mkt": "en-US"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                log.info("Bing RSS response HTTP %s", resp.status_code)
                return []
            items = parse_rss(resp.content, "Bing web-search RSS", 10)
            log.info("Bing RSS parsed %d metadata results", len(items))
            return items

        # Independent official sources are discovered through the same free
        # Bing RSS endpoint, without constructing or guessing document URLs.
        # One short site-scoped lookup per organization, max three in total.
        organizations = [
            org for org in OFFICIAL_SCIENCE_SITES
            if org in terms
        ][:3]
        subjects = [x for x in terms if x in SCIENCE_TOPIC_WORDS][:5]
        async def get_official_site(org: str):
            host = OFFICIAL_SCIENCE_SITES[org][0]
            q = "site:" + host + " " + " ".join(subjects or terms[:4])
            resp = await session.get(
                "https://www.bing.com/search",
                params={"q": q, "format": "rss", "mkt": "en-US"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                log.info("Official institution RSS unavailable org=%s status=%s",
                         org, resp.status_code)
                return []
            items = parse_rss(resp.content, "Official site discovery via Bing RSS", 6)
            selected = [i for i in items
                        if urlsplit(i["url"]).hostname in OFFICIAL_SCIENCE_SITES[org]
                        and not urlsplit(i["url"]).path.lower().endswith((".pdf", ".xml", ".json"))]
            log.info("Official institution site discovery org=%s candidates=%d",
                     org, len(selected))
            return selected[:4]

        # Bound extra discovery to explicitly requested organizations, without
        # hardcoding any article URL or following search-result links here.
        mentioned = query_text.casefold()
        accessibility_orgs = [
            name for name in ACCESSIBILITY_SOURCE_HOSTS
            if (name in ("w3c", "mdn") and re.search(
                r"\b" + re.escape(name) + r"\b", mentioned
            ) or name == "govuk" and re.search(r"\bgov\.?\s*uk\b", mentioned))
        ][:3]
        accessibility_subject = [
            word for word in terms if word in ACCESSIBILITY_TOPIC_WORDS
        ][:5]
        async def get_accessibility_site(org: str):
            host = ACCESSIBILITY_SOURCE_HOSTS[org]
            q = "site:" + host + " " + " ".join(
                accessibility_subject or terms[:4]
            )
            resp = await session.get(
                "https://www.bing.com/search",
                params={"q": q, "format": "rss", "mkt": "en-US"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                return []
            found = parse_rss(resp.content, "Institution-scoped Bing RSS", 6)
            result = [
                item for item in found
                if urlsplit(item["url"]).hostname == host and
                not urlsplit(item["url"]).path.lower().endswith(
                    (".pdf", ".xml", ".json")
                )
            ]
            log.info("Accessibility institution discovery org=%s candidates=%d",
                     org, len(result))
            return result[:4]

        async def get_wikipedia():
            resp = await session.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query,
                        "srlimit": "5", "format": "json"},
                timeout=7.0,
            )
            if resp.status_code != 200 or len(resp.content) > MAX_BYTES:
                log.info("Wikipedia API response HTTP %s", resp.status_code)
                return []
            items = parse_wikipedia(resp.json(), 5)
            log.info("Wikipedia API parsed %d results", len(items))
            return items

        async def get_news():
            resp = await session.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                log.info("Google News RSS response HTTP %s", resp.status_code)
                return []
            items = parse_rss(resp.content, "Google News RSS (headline)", 5)
            log.info("Google News RSS parsed %d metadata results", len(items))
            return items

        async def get_stackexchange():
            resp = await session.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={"q": technical_query, "site": "stackoverflow",
                        "sort": "relevance", "order": "desc", "pagesize": "8"},
                timeout=8.0,
            )
            if resp.status_code != 200 or len(resp.content) > MAX_BYTES:
                log.info("Stack Overflow API response HTTP %s", resp.status_code)
                return []
            try:
                data = resp.json()
            except ValueError:
                return []
            if data.get("backoff"):
                log.info("Stack Exchange returned backoff; no retries performed")
            items = parse_stackexchange(data, 8)
            log.info("Stack Overflow API parsed %d metadata results", len(items))
            return items

        async def get_hackernews():
            resp = await session.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": technical_query, "tags": "story", "hitsPerPage": "6"},
                timeout=8.0,
            )
            if resp.status_code != 200 or len(resp.content) > MAX_BYTES:
                log.info("Hacker News API response HTTP %s", resp.status_code)
                return []
            try:
                data = resp.json()
            except ValueError:
                return []
            items = parse_hackernews(data, 6)
            log.info("Hacker News API parsed %d metadata results", len(items))
            return items

        providers = [get_bing(), get_wikipedia(), get_news()]
        providers.extend(get_official_site(org) for org in organizations)
        providers.extend(get_accessibility_site(org) for org in accessibility_orgs)
        if tech:
            # Relevant community references are helpful for programming questions.
            # Not used as evidence of official API behavior.
            providers.extend([get_stackexchange(), get_hackernews()])
        results = await asyncio.gather(*providers, return_exceptions=True)
        ranked, urls = [], set()
        for result in results:
            if isinstance(result, BaseException):
                log.warning("Public source feed unavailable: %s", type(result).__name__)
                continue
            for item in result:
                if item["url"] in urls:
                    continue
                urls.add(item["url"])
                score = source_relevance(item, terms)
                if score > 0:
                    ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        # General open-web reading is limited to three network attempts: put
        # directly readable article-style URLs before tracking/news redirects.
        # Preserve institution-specific ranking for the curated protocols.
        if not organizations and not accessibility_orgs:
            ranked = prioritize_readable_sources(ranked)
        # Prefer one strongly-relevant article from each requested institution,
        # instead of three near-duplicate pages from a single organization.
        if len(organizations) >= 2 or len(accessibility_orgs) >= 2:
            institutional, used_urls = [], set()
            target_hosts = (
                [OFFICIAL_SCIENCE_SITES[name] for name in organizations]
                + [(ACCESSIBILITY_SOURCE_HOSTS[name],)
                   for name in accessibility_orgs]
            )
            for hosts in target_hosts:
                for _score, item in ranked:
                    if (urlsplit(item["url"]).hostname in hosts
                            and item["url"] not in used_urls):
                        institutional.append((_score, item))
                        used_urls.add(item["url"])
                        break
            ranked = institutional + [
                pair for pair in ranked if pair[1]["url"] not in used_urls
            ]
        log.info("Public source search selected %d of %d unique results (technical=%s)",
                 min(len(ranked), MAX_SOURCES), len(urls), tech)
        return [item for _score, item in ranked[:MAX_SOURCES]]

    if client is not None:
        return await search(client)
    async with httpx.AsyncClient(headers=WEB_HEADERS, follow_redirects=True) as new_client:
        return await search(new_client)


def evidence_text(sources: list[dict]) -> str:
    """A clear data boundary. The text is search metadata and must not be obeyed."""
    chunks = []
    for i, source in enumerate(sources[:MAX_SOURCES], start=1):
        read = source.get("read_status") == "verified_page_excerpt" and bool(source.get("excerpt"))
        parts = [
            f"[{i}] Provider: {clean(source['provider'], 50)}",
            "Title: " + clean(source["title"], 115),
            "URL: " + source["url"],
            "Evidence type: FETCHED PAGE EXCERPT (selected text, not entire page)"
            if read else ("Evidence type: OFFICIAL PAGE FETCH FAILED (not read)"
                          if source.get("read_status") == "page_unavailable"
                          else "Evidence type: SEARCH METADATA ONLY (page not read)"),
        ]
        if read:
            parts.append("Untrusted page excerpt: " + clean(source["excerpt"], 2400))
        elif source.get("read_status") == "page_unavailable":
            parts.append("Note: No page text was retrieved or quoted.")
        else:
            parts.append("Snippet/headline only: " + clean(source.get("snippet", ""), 300))
        if source.get("published"):
            parts.append("Feed date: " + clean(source["published"], 55))
        chunks.append("\n".join(parts))
    return "\n\n".join(chunks)


def references_text(sources: list[dict]) -> str:
    return "\n\nSenarai sumber (petikan terpilih atau metadata; bukan semakan artikel penuh):\n" + "\n".join(
        f"[{i}] {clean(source['title'], 95)} "
        + ("[Petikan halaman diambil]" if source.get("read_status") == "verified_page_excerpt"
           and source.get("excerpt") else ("[Gagal diakses; halaman tidak dibaca]"
            if source.get("read_status") == "page_unavailable" else "[Metadata sahaja]"))
        + f"\n{source['url']}"
        for i, source in enumerate(sources[:MAX_SOURCES], start=1)
    )
