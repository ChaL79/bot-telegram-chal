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
from urllib.parse import parse_qs, quote, urlsplit
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
# Reviewed domains for explicit brand-scoped discovery; no article URLs guessed.
PRODUCT_OFFICIAL_DOMAINS = {
    "kingston": ("www.kingston.com",),
    "crucial": ("www.crucial.com",),
    "samsung": ("semiconductor.samsung.com",),
}

# Sitemap INDEX endpoints independently checked against manufacturer robots.txt.
# Never treat a sitemap URL as article content; each page must be fetched.
PRODUCT_OFFICIAL_SITEMAPS = {
    "samsung": "https://semiconductor.samsung.com/sitemap.xml",
}
MAX_SITEMAP_BYTES = 350_000

# Marketplace product offers are not technical documentation evidence.
RESEARCH_MARKETPLACE_HOSTS = frozenset({
    "woot.com", "amazon.com", "ebay.com", "bestbuy.com", "newegg.com",
})


def is_marketplace_listing(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == base or host.endswith("." + base)
               for base in RESEARCH_MARKETPLACE_HOSTS)



def requested_product_institutions(question: str) -> list[str]:
    return [name for name in PRODUCT_OFFICIAL_DOMAINS
            if re.search(r"\b" + re.escape(name) + r"\b", question, re.I)][:3]


def requested_ssd_manufacturers(question: str) -> list[str]:
    """Return explicitly requested manufacturers for NVMe/SATA SSD comparisons."""
    terms = set(search_query_terms(question))
    if not {"ssd", "nvme", "sata"}.issubset(terms):
        return []
    return requested_product_institutions(question)


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
    "bagi", "mereka", "amerika", "syarikat", "united", "kingdom",
    "penjelasan", "cadangan", "kesannya", "dalamnya",
})


# Bilingual, bounded concept matching for English-language official pages.
# Maps Malay subject terms only; it does not translate entire user documents,
# issue paid model calls, or bypass any relevance/security check.
MALAY_RESEARCH_TERMS = {
    "pengurusan": "management",
    "keletihan": "fatigue",
    "pekerja": "workers",
    "kerja": "work",
    "syif": "shift",
    "tidur": "sleep",
    "kekurangan": "lack",
    "keselamatan": "safety",
    "risiko": "risk",
    "pencegahan": "prevention",
    "malam": "night",
    "kemalangan": "accidents",
    "kecederaan": "injuries",
    "kesihatan": "health",
    # Cross-domain maritime and satellite concepts: vocabulary only, no vendor URLs.
    "satelit": "satellite",
    "satelitnya": "satellite",
    "kapal": "maritime",
    "laut": "maritime",
    "kelajuan": "speed",
    "liputan": "coverage",
    "cuaca": "weather",
    "hujan": "rain",
    "pendaman": "latency",
    "peralihan": "handover",
    "kestabilan": "reliability",
    # Broad Malay -> English concepts for general-interest research.
    "pendidikan": "education", "sekolah": "school", "universiti": "university",
    "ekonomi": "economy", "perniagaan": "business", "harga": "price",
    "pasaran": "market", "pekerjaan": "employment", "gaji": "salary",
    "pelancongan": "tourism", "sejarah": "history",
    "budaya": "culture", "agama": "religion", "islam": "islam",
    "teknologi": "technology", "komputer": "computer",
    "kesan": "effects", "faedah": "benefits", "kebaikan": "benefits",
    "keburukan": "risks", "masalah": "problems", "punca": "causes",
    "makanan": "food", "pemakanan": "nutrition", "senaman": "exercise",
    "persekitaran": "environment", "pencemaran": "pollution",
    "tenaga": "energy", "elektrik": "electricity", "kos": "cost",
    "telefon": "phone", "pembelajaran": "learning",
    "alam": "nature", "semula": "natural",
    "kenaikan": "rise", "paras": "level", "pencairan": "melting",
    "glasier": "glaciers", "unjuran": "projection",
    "ketidakpastian": "uncertainty", "pemanasan": "warming",
    "lautan": "ocean", "kajian": "research",
    "perkakasan": "hardware", "privasi": "privacy",
    "ketepatan": "accuracy", "perbandingan": "comparison",
}

def _terms(value: str) -> list[str]:
    # Keep hyphenated product names and their version as one search concept.
    # "Wi-Fi 7", "WiFi7", and "WiFi 7" all get wifi7 + wifi; the same
    # normalization is applied to titles/snippets when scoring RSS hits.
    value = value.casefold()
    value = re.sub(r"\balam\s+sekitar\b", "persekitaran", value)
    # Malay climate terminology maps to established English scientific anchors.
    # Use before tokenization; a direct NOAA/NASA/IPCC comparison must not
    # collapse into unrelated agency landing pages.
    value = re.sub(r"\bparas\s+laut\b", "sea level", value)
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
    "noaa", "nasa", "ipcc", "sea", "level", "rise", "projection", "uncertainty",
    "thermal", "expansion", "ocean", "glaciers", "glacier",
    "ice", "sheets", "melting", "climate",
    "wcag", "accessibility", "keyboard", "focus", "visibility",
    "obscured", "w3c", "mdn", "govuk",
    # Cross-domain consumer/technical research; not a vendor URL allowlist.
    "wifi7", "wifi6", "wifi", "mlo", "6ghz", "latency", "multilink",
    # Occupational safety: keep the subject, not Malay task scaffolding, first.
    "fatigue", "shift", "work", "sleep", "cdc", "niosh", "hse",
    # Orbit terminology must outrank generic latency/speed in comparison queries.
    "leo", "geo", "satellite", "maritime", "vsat",
    "handover", "orbit", "geostationary", "constellation",
})
IDENTITY_ANCHORS = frozenset({
    "passkeys", "passkey", "fido2", "totp", "mfa", "otp", "phishing",
    "authenticator", "webauthn", "sms",
})


def search_query_terms(request: str) -> list[str]:
    """Extract relevant terms across a complete question; no LLM, no extra API."""
    request = clean(request, 2800)
    request = re.sub(r"^/(?:deepdive|power)\b\s*", "", request, flags=re.I)
    words = [MALAY_RESEARCH_TERMS.get(x, x) for x in _terms(request)
             if not x.isdecimal()]
    # "Kekurangan tidur pekerja" is an occupational fatigue question even
    # when the user never uses the word keletihan/fatigue. Add the conceptual
    # anchor before ranking; do not make all generic sleep topics fatigue.
    if ({"sleep", "lack"}.issubset(words)
            and {"workers", "work", "offshore", "shift"} & set(words)
            and "fatigue" not in words):
        words.insert(0, "fatigue")
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
    if term in ("satellite", "satellites"):
        return "satellite"
    if term in ("handover", "handovers", "handoff", "handoffs"):
        return "handover"
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
    # Require the defining subject, not incidental performance terminology.
    # A page about generic network latency cannot establish LEO/GEO evidence.
    # This is a topic-based rule; no vendor, hostname or article URL is preferred.
    if ({"leo", "geo"} & set(sought)) and "satellite" in sought:
        present = title_terms | content_terms
        orbit_context = {"satellite", "orbit", "orbital", "maritime",
                         "vsat", "geostationary", "constellation"}
        applied_context = {"geo", "geostationary", "orbit", "orbital", "maritime",
                           "offshore", "vessel", "ship", "handover",
                           "coverage", "rain", "weather", "broadband"}
        if (not (present & {"leo", "geo"})
                or not (present & orbit_context)
                or not (present & applied_context)):
            return 0
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
    score = len(matched) + 2 * len(matched & title_terms)
    # Source-specific keywords from the user's request beat incidental matching:
    # generic LEO/GEO press releases must not displace offshore/maritime evidence.
    # This boosts the query SUBJECT (not publishers or static article URLs).
    if "maritime" in sought and ({"leo", "geo"} & set(sought)):
        if {"maritime", "offshore", "vessel", "ship", "ships"} & title_terms:
            score += 8
        elif {"maritime", "offshore", "vessel", "ship", "ships"} & content_terms:
            score += 3
    return score


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


def parse_bing_news_rss(raw: bytes, *, limit: int = 8) -> list[dict]:
    """Extract actual publisher URLs from Bing News RSS metadata without following redirects.

    Search providers can wrap a publisher URL inside a Bing /news/apiclick.aspx
    link. Only extract the single encoded `url` parameter of that exact endpoint.
    The extracted URL is still UNVERIFIED until the independent DNS-pinned HTML
    reader fetches and extracts real relevant page content.
    """
    if len(raw) > MAX_BYTES or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        return []
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return []
    found = []
    used = set()
    for item in root.findall("./channel/item")[:limit * 5]:
        wrapper = (item.findtext("link") or "").strip()
        try:
            p = urlsplit(wrapper)
            if (p.scheme not in ("http", "https")
                    or p.hostname not in ("www.bing.com", "bing.com")
                    or p.path.lower() != "/news/apiclick.aspx"
                    or p.username or p.password or p.fragment):
                continue
            candidates = parse_qs(p.query, keep_blank_values=False).get("url", [])
            if len(candidates) != 1:
                continue
            target = candidates[0]
            if target in used or not safe_public_article_url(target):
                continue
        except (UnicodeError, ValueError):
            continue
        title = clean(item.findtext("title") or "", 115)
        if not title:
            continue
        used.add(target)
        found.append({
            "provider": "Bing News RSS (publisher link; HTML not yet verified)",
            "title": title, "url": target,
            "snippet": clean(item.findtext("description") or "", 300),
            "published": clean(item.findtext("pubDate") or "", 55),
        })
        if len(found) >= limit:
            break
    return found


def parse_wikipedia(data: dict, limit: int = 3, *, language: str = "en") -> list[dict]:
    if language not in ("en", "ms") or not isinstance(data, dict):
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
        url = f"https://{language}.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_-")
        if allowed_source_url(url):
            result.append({"provider": f"Wikipedia ({language})", "title": title, "url": url,
                           "snippet": clean(hit.get("snippet", ""), 300),
                           "published": ""})
        if len(result) == limit:
            break
    return result


def canonical_article_key(url: str) -> tuple[str, str, str, str]:
    """Deduplicate same-host AMP aliases without fetching or rewriting URLs."""
    parsed = urlsplit(url)
    parts = (parsed.path or "/").split("/")
    if len(parts) > 2 and parts[1].casefold() == "amp":
        parts.pop(1)
    path = "/".join(parts).rstrip("/") or "/"
    return (parsed.scheme.casefold(), (parsed.hostname or "").casefold(),
            path, parsed.query)


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
    named_product_orgs = requested_ssd_manufacturers(query_text)
    subject_terms = [word for word in terms if word not in named_product_orgs]
    query = " ".join((subject_terms[:4] if len(named_product_orgs) >= 1
                      and len(subject_terms) >= 2 else terms[:6]))[:160]

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

        async def get_bing_short():
            # A verbose multi-condition search may return generic/noisy results.
            # A second, independent concise query makes discovery less brittle
            # without using a paid API or broadening the reader's URL policy.
            key_terms = [term for term in terms if len(term) >= 5 and term not in {"wifi"}]
            short_query = (" ".join([term for term in ("nvme", "sata", "ssd") if term in subject_terms][:2])
                           if len(named_product_orgs) >= 1
                           else " ".join(key_terms[:2])[:80])
            if (organizations or accessibility_orgs or len(terms) <= 5
                    or not short_query or short_query == query):
                return []
            resp = await session.get(
                "https://www.bing.com/search",
                params={"q": short_query, "format": "rss", "mkt": "en-US"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                log.info("Bing concise RSS response HTTP %s", resp.status_code)
                return []
            items = parse_rss(resp.content, "Bing concise web-search RSS", 10)
            log.info("Bing concise RSS parsed %d metadata results", len(items))
            return items

        async def get_product_site(org: str) -> list[dict]:
            # Bing RSS can ignore site: and return unrelated retail results.
            # Use subject-led named-maker search, then enforce exact host matching.
            q_terms = [term for term in ("nvme", "sata", "ssd", "pcie")
                       if term in subject_terms]
            q = " ".join((q_terms[:3] or subject_terms[:3]) + [org])
            resp = await session.get("https://www.bing.com/search",
                                     params={"q": q, "format": "rss", "mkt": "en-US"},
                                     timeout=7.0)
            if resp.status_code != 200:
                return []
            items = parse_rss(resp.content, "Official product institution via Bing RSS", 10)
            found = [item for item in items
                     if urlsplit(item["url"]).hostname in PRODUCT_OFFICIAL_DOMAINS[org]
                     and not urlsplit(item["url"]).path.casefold().endswith((".pdf", ".xml", ".json"))]
            log.info("Product official discovery org=%s candidates=%d", org, len(found))
            return found[:4]

        async def get_product_sitemap(org: str) -> list[dict]:
            """Read a fixed manufacturer sitemap, never arbitrary user XML URLs."""
            endpoint = PRODUCT_OFFICIAL_SITEMAPS.get(org)
            if not endpoint:
                return []
            expected_host = PRODUCT_OFFICIAL_DOMAINS[org][0]
            if urlsplit(endpoint).hostname != expected_host:
                return []
            data = bytearray()
            try:
                async with session.stream(
                    "GET", endpoint, follow_redirects=False,
                    headers={"Accept": "application/xml,text/xml"},
                    timeout=7.0,
                ) as response:
                    if response.status_code != 200:
                        return []
                    ctype = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if ctype not in ("application/xml", "text/xml", "text/plain", "application/octet-stream"):
                        return []
                    length = response.headers.get("content-length", "")
                    if length.isdecimal() and int(length) > MAX_SITEMAP_BYTES:
                        return []
                    async for part in response.aiter_bytes():
                        data.extend(part)
                        if len(data) > MAX_SITEMAP_BYTES:
                            return []
            except (httpx.HTTPError, OSError, ValueError):
                return []
            if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
                return []
            try:
                root = ElementTree.fromstring(bytes(data))
            except ElementTree.ParseError:
                return []
            candidates = []
            for node in root.iter():
                if not node.tag.endswith("}loc") and node.tag != "loc":
                    continue
                url = (node.text or "").strip()
                if not safe_public_article_url(url):
                    continue
                parsed = urlsplit(url)
                if parsed.hostname != expected_host:
                    continue
                path = parsed.path.casefold()
                if not any(x in path for x in ("ssd", "nvme", "sata")):
                    continue
                relevance = (70 if "internalssd-product-information" in path else
                             55 if "nvme" in path or "sata" in path else
                             40 if "/support/faqs/internalssd" in path else
                             20 if "/support/" in path else 5)
                candidates.append((relevance, {
                    "provider": "Official manufacturer sitemap URL (page not read)",
                    "title": org.capitalize() + " " + parsed.path.rstrip("/").split("/")[-1].replace("-", " "),
                    "url": url, "snippet": "", "published": "",
                }))
                if len(candidates) >= 300:
                    break
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            log.info("Manufacturer sitemap org=%s relevant_urls=%d",
                     org, len(candidates))
            return [item for _, item in candidates[:3]]

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

        async def get_bing_news():
            """Bounded universal news fallback; no vendor URLs or redirect fetches.

            Search abbreviations are ambiguous (e.g. LEO -> zodiac). A second
            reordered phrase is tried only if the first has too few strongly
            topical publisher domains. This is generic query planning, not a
            satellite-domain allowlist.
            """
            primary = " ".join(terms[:6])[:160]
            acronyms = [t for t in terms if len(t) <= 3][:2]
            subjects = [t for t in terms if len(t) > 3][:2]
            alternative = " ".join(acronyms + subjects)[:160]
            queries = [primary]
            if acronyms and alternative and alternative != primary:
                queries.append(alternative)
            gathered = []
            unique_urls = set()
            for search_phrase in queries[:2]:
                resp = await session.get(
                    "https://www.bing.com/news/search",
                    params={"q": search_phrase, "format": "rss"},
                    timeout=7.0,
                    follow_redirects=False,
                )
                if resp.status_code != 200:
                    log.info("Bing News RSS unavailable HTTP %s", resp.status_code)
                    continue
                items = parse_bing_news_rss(resp.content, limit=12)
                log.info("Bing News publisher links discovered %d", len(items))
                for item in items:
                    if item["url"] not in unique_urls:
                        unique_urls.add(item["url"])
                        gathered.append(item)
                topical_hosts = {
                    urlsplit(item["url"]).hostname for item in gathered
                    if source_relevance(item, terms) > 0
                }
                if len(topical_hosts) >= 4:
                    break
            return gathered

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

        async def get_wikipedia_malay():
            # Community encyclopedia search: metadata only, never treat a
            # Wikipedia search snippet as verified article contents.
            short = " ".join(_terms(query_text)[:7])[:110]
            if not short:
                return []
            resp = await session.get(
                "https://ms.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": short,
                        "srlimit": "4", "format": "json"},
                timeout=7.0,
            )
            if resp.status_code != 200 or len(resp.content) > MAX_BYTES:
                return []
            return parse_wikipedia(resp.json(), 4, language="ms")

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

        providers = [get_bing(), get_bing_short(), get_wikipedia(),
                     get_wikipedia_malay(), get_news()]
        if len(named_product_orgs) >= 1:
            providers.extend(get_product_site(org) for org in named_product_orgs)
            providers.extend(get_product_sitemap(org) for org in named_product_orgs
                             if org in PRODUCT_OFFICIAL_SITEMAPS)
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
                # Explicit documentation comparisons must not be supported by
                # storefront listings, even when the title mentions NVMe/SATA.
                if (len(named_product_orgs) >= 1
                        and re.search(r"\b(?:rasmi|dokumentasi|documentation|official)\b",
                                      query_text, re.I)
                        and is_marketplace_listing(item["url"])):
                    continue
                score = source_relevance(item, terms)
                # Trusted sitemap path proves discovery of a URL, NOT page content.
                # It is eligible for article reading only in an explicitly named
                # maker/SSD comparison and must still pass the pinned HTML reader.
                if (item.get("provider") ==
                        "Official manufacturer sitemap URL (page not read)"
                        and len(named_product_orgs) >= 1
                        and "ssd" in terms
                        and any(urlsplit(item["url"]).hostname in PRODUCT_OFFICIAL_DOMAINS[org]
                                for org in named_product_orgs)
                        and safe_public_article_url(item["url"])):
                    score = max(score, 6)
                if score > 0:
                    ranked.append((score, item))
        # Invoke the independent news RSS index only when primary providers
        # failed to discover two different directly-readable publisher hosts.
        # Metadata wrappers and encyclopedias are not counted as evidence here.
        weak_hosts = {"news.google.com", "en.wikipedia.org",
                      "en.m.wikipedia.org", "ms.wikipedia.org"}
        genuine_hosts = {
            urlsplit(item["url"]).hostname for _, item in ranked
            if safe_public_article_url(item["url"])
            and urlsplit(item["url"]).hostname not in weak_hosts
        }
        if len(genuine_hosts) < 2:
            try:
                news_hits = await get_bing_news()
            except (httpx.HTTPError, ValueError, UnicodeError) as exc:
                log.info("Bing News RSS discovery unavailable kind=%s",
                         type(exc).__name__)
                news_hits = []
            for item in news_hits:
                if item["url"] in urls:
                    continue
                urls.add(item["url"])
                score = source_relevance(item, terms)
                if score > 0:
                    ranked.append((score, item))
            log.info("Independent news discovery candidates=%d",
                     sum(1 for _, ref in ranked
                         if ref.get("provider", "").startswith("Bing News RSS")))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        # General open-web reading is limited to three network attempts: put
        # directly readable article-style URLs before tracking/news redirects.
        # Preserve institution-specific ranking for the curated protocols.
        if not organizations and not accessibility_orgs:
            if "maritime" in terms and "satellite" in terms:
                # Prefer evidence explicitly about the requested maritime
                # context before generic press releases with deep URL paths.
                ranked = sorted(
                    ranked,
                    key=lambda pair: (pair[0], readable_article_priority(pair[1])),
                    reverse=True,
                )
                selected, repeats, hosts = [], [], set()
                for pair in ranked:
                    hostname = urlsplit(pair[1]["url"]).hostname
                    if hostname in hosts:
                        repeats.append(pair)
                    else:
                        hosts.add(hostname)
                        selected.append(pair)
                ranked = selected + repeats
            else:
                ranked = prioritize_readable_sources(ranked)
        # Preserve distinct official sources before generic indexing results.
        if len(named_product_orgs) >= 1:
            official, selected_urls = [], set()
            for org in named_product_orgs:
                for pair in ranked:
                    if (urlsplit(pair[1]["url"]).hostname in PRODUCT_OFFICIAL_DOMAINS[org]
                            and pair[1]["url"] not in selected_urls):
                        official.append(pair)
                        selected_urls.add(pair[1]["url"])
                        break
            ranked = official + [pair for pair in ranked
                                 if pair[1]["url"] not in selected_urls]
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
        # AMP /amp/news/... and conventional /news/... paths can describe
        # the same article. This is discovery de-duplication, NOT verification.
        canonical_seen, unique_ranked = set(), []
        for pair in ranked:
            key = canonical_article_key(pair[1]["url"])
            if key in canonical_seen:
                continue
            canonical_seen.add(key)
            unique_ranked.append(pair)
        log.info("Public source search selected %d of %d unique results (technical=%s)",
                 min(len(unique_ranked), MAX_SOURCES), len(urls), tech)
        return [item for _score, item in unique_ranked[:MAX_SOURCES]]

    if client is not None:
        return await search(client)
    async with httpx.AsyncClient(headers=WEB_HEADERS, follow_redirects=True) as new_client:
        return await search(new_client)


def evidence_text(sources: list[dict]) -> str:
    """Untrusted source data with truthful read-attempt provenance."""
    chunks = []
    for i, source in enumerate(sources[:MAX_SOURCES], start=1):
        status = source.get("read_status", "search_metadata_only")
        read = status == "verified_page_excerpt" and bool(source.get("excerpt"))
        if read:
            evidence_type = "FETCHED PAGE EXCERPT (selected text, not entire page)"
        elif status == "page_unavailable":
            evidence_type = "PAGE READ ATTEMPTED; NO USABLE EXCERPT (not verified)"
        elif status == "source_not_found":
            evidence_type = "REQUESTED OFFICIAL SOURCE NOT FOUND IN SEARCH (no URL, no fetch)"
        else:
            evidence_type = "SEARCH METADATA ONLY (page not read)"
        parts = [
            f"[{i}] Provider: {clean(source['provider'], 50)}",
            "Title: " + clean(source["title"], 115),
            "URL: " + (source.get("url") or "None discovered"),
            "Evidence type: " + evidence_type,
        ]
        if read:
            parts.append("Untrusted page excerpt: " + clean(source["excerpt"], 2400))
        elif status == "page_unavailable":
            parts.append("Note: A page read was attempted but no usable excerpt was obtained.")
        elif status == "source_not_found":
            parts.append("Note: No relevant official URL was discovered and no read was attempted.")
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
           and source.get("excerpt") else ("[Cubaan baca gagal; tiada petikan sah]"
            if source.get("read_status") == "page_unavailable" else
            ("[Sumber rasmi tidak ditemui; belum dicuba]"
             if source.get("read_status") == "source_not_found"
             else "[Metadata sahaja; belum dibaca]")))
        + f"\n{source.get('url') or 'URL rasmi tidak ditemui dalam carian'}"
        for i, source in enumerate(sources[:MAX_SOURCES], start=1)
    )
