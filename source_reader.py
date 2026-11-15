"""Strictly bounded read-only source excerpts for /deepdive.

Only pre-approved public documentation or Stack Overflow question pages can
be fetched. Arbitrary result URLs are never requested. Unapproved redirects
are rejected, including redirects to private or unrelated destinations.
Returned text is UNTRUSTED evidence, not a source of instructions.
"""
import asyncio
import html
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin
from typing import Optional

import httpx
from free_research import clean, search_query_terms

log = logging.getLogger("source_reader")
MAX_PAGE_BYTES = 450_000
MAX_EXCERPT = 2400
MAX_FETCHES = 3
MAX_GOOGLE_FETCHES = 9  # fixed official documentation only, not general crawling
FETCH_TIMEOUT = 7.0

OFFICIAL_ASYNCIO = "https://docs.python.org/3/library/asyncio.html"
OFFICIAL_SEMAPHORE = "https://docs.python.org/3/library/asyncio-sync.html"
AIOLIMITER_DOCS = "https://aiolimiter.readthedocs.io/en/stable/"

# Fixed public pages for authentication security research. Do not discover
# arbitrary links from untrusted RSS metadata or permit cross-host redirects.
IDENTITY_DOCS = (
    ("https://pages.nist.gov/800-63-4/sp800-63b/authenticators/",
     "NIST SP 800-63B: Authenticator and Verifier Requirements",
     "NIST phishing resistance OTP authenticator cryptographic passkeys"),
    ("https://fidoalliance.org/passkeys-2/",
     "FIDO Alliance: Passkeys",
     "FIDO Alliance passkeys FIDO2 phishing-resistant public key authentication"),
    ("https://www.cisa.gov/MFA",
     "CISA: Multi-Factor Authentication",
     "CISA phishing resistant MFA SMS TOTP SIM swapping"),
)
IDENTITY_ALLOWED_URLS = frozenset(x[0] for x in IDENTITY_DOCS)


def identity_official_docs(question: str) -> list[dict]:
    words = set(search_query_terms(question))
    if not words.intersection({"passkeys", "passkey", "fido2", "totp",
                               "mfa", "phishing", "webauthn", "authenticator"}):
        return []
    return [
        {"provider": "Official authentication security guidance", "title": title,
         "url": url, "snippet": "", "published": "", "focus": focus}
        for url, title, focus in IDENTITY_DOCS
    ]

# Official, public Google pages. Exact URLs only; no private account pages,
# arbitrary paths, user-provided URLs, query parameters or automatic paid APIs.
GOOGLE_DOCS = {
    "gemini_available": ("https://support.google.com/gemini/answer/13575153",
                         "Gemini Apps supported countries", "Gemini web app supported countries Brunei"),
    "gemini_overview": ("https://blog.google/innovation-and-ai/products/google-ai-updates-may-2025/",
                        "Google Gemini app feature overview (June 2025)", "Gemini app Deep Research Canvas Gemini Live features"),
    "studio_start": ("https://aistudio.google.com/docs/get-started",
                     "Google AI Studio and Gemini API getting started",
                     "Google AI Studio Gemini API key Free Tier prompts models SDK"),
    "studio_free": ("https://aistudio.google.com/docs/pricing",
                    "Gemini API free and paid pricing",
                    "Gemini API Free Tier limited models paid pricing"),
    "api_terms": ("https://aistudio.google.com/docs/terms.md",
                  "Gemini API Additional Terms - Unpaid and Paid data use",
                  "Gemini API Unpaid Services Paid Services privacy use human review"),
    "antigravity_guide": ("https://antigravity.google/docs/getting-started",
                          "Google Antigravity getting started", "Antigravity coding agents IDE CLI"),
    "antigravity_free": ("https://antigravity.google/pricing",
                         "Google Antigravity individual pricing", "Antigravity free individual plan rate limits"),
    "notebook_rename": ("https://blog.google/innovation-and-ai/products/gemini-notebook/notebooklm-gemini-notebook/",
                        "Google NotebookLM renamed Gemini Notebook", "NotebookLM Gemini Notebook rename research"),
    "flow_intro": ("https://blog.google/innovation-and-ai/products/google-flow-veo-ai-filmmaking-tool/",
                    "Google Flow video creation overview (May 2025)", "Flow AI filmmaking Veo Imagen Gemini scenes video"),
    "flow_available": ("https://support.google.com/flow/answer/16353544",
                       "Google Flow supported countries", "Google Flow countries Brunei availability"),
}
GOOGLE_ALLOWED_URLS = frozenset(item[0] for item in GOOGLE_DOCS.values())


def google_official_docs(question: str) -> list[dict]:
    """Select relevant exact official Google sources by product name, not guessed URLs."""
    q = question.casefold()
    google_context = any(x in q for x in ("google", "gemini", "antigravity", "notebooklm"))
    if not google_context:
        return []
    requested = []
    # Gemini API privacy inquiries must use the contract governing data use,
    # not just generic Google News or the consumer Gemini privacy policy.
    privacy_topic = bool(re.search(
        r"\b(?:privacy|confidential|sensitive|personal data|data use|training|"
        r"train|human review|improve.*products|dasar privasi|data sulit)\b", q
    ))
    api_context = "gemini api" in q or "gemini" in q and "api" in q
    if api_context and privacy_topic:
        requested.extend(["api_terms", "studio_free"])
    # "Google AI Studio" is a developer product, NOT a request to research the
    # consumer Gemini app. Avoid spending the first source slots on wrong products.
    consumer_gemini = any(x in q for x in ("gemini ai", "gemini app", "gemini assistant"))
    if consumer_gemini:
        requested.append("gemini_available")
        # For a single-product query an overview adds value; for a five-product
        # comparison, prefer accessibility and pricing references for all five.
        product_count = sum(bool(x in q) for x in
            ("ai studio", "antigravity", "notebooklm", "flow"))
        if product_count <= 2:
            requested.append("gemini_overview")
    if "ai studio" in q or "aistudio" in q:
        requested.extend(["studio_start", "studio_free"])
    if "antigravity" in q:
        requested.extend(["antigravity_guide", "antigravity_free"])
    if "notebooklm" in q or "notebook lm" in q or "gemini notebook" in q:
        requested.append("notebook_rename")
    if re.search(r"\bflow\b", q):
        requested.extend(["flow_intro", "flow_available"])
    result = []
    for k in dict.fromkeys(requested):
        url, title, focus = GOOGLE_DOCS[k]
        result.append({"provider": "Official Google documentation", "title": title,
                       "url": url, "snippet": "", "published": "", "focus": focus})
    return result



def is_approved_document_url(url: str) -> bool:
    """Avoid SSRF by fetching only exact hosts and tightly scoped HTTPS paths."""
    if not isinstance(url, str) or len(url) > 700 or any(c in url for c in "\r\n\t<>"):
        return False
    try:
        p = urlsplit(url)
        if p.scheme != "https" or p.port not in (None, 443) or p.username or p.password:
            return False
        host, path = p.hostname or "", p.path
        if url in GOOGLE_ALLOWED_URLS or url in IDENTITY_ALLOWED_URLS:
            return True
        # Some Google developer docs canonicalize to an explicit English locale.
        # Allow only that locale on an exactly approved ai.google.dev path.
        if host == "ai.google.dev" and not p.fragment and p.query == "hl=en":
            for base in GOOGLE_ALLOWED_URLS:
                b = urlsplit(base)
                if b.hostname == host and b.path.rstrip("/") == path.rstrip("/"):
                    return True
        if host in ("support.google.com", "ai.google.dev", "aistudio.google.com",
                    "antigravity.google", "blog.google"):
            return False
        if host == "docs.python.org":
            return bool(re.fullmatch(r"/3(?:\.\d+)?/(?:library/[a-z0-9_.-]+\.html)?", path))
        if host == "stackoverflow.com":
            return bool(re.fullmatch(r"/questions/[0-9]+(?:/[a-zA-Z0-9_%-]+)?/?", path))
        if host == "aiolimiter.readthedocs.io":
            return path in ("/en/stable/", "/en/stable/index.html")
        return False
    except ValueError:
        return False


class MainText(HTMLParser):
    """Extract text in HTML <main>, <article> or an equivalent document body."""
    BLOCKS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "pre", "dt", "dd",
                         "tr", "div", "section", "br", "blockquote"})
    VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                      "link", "meta", "param", "source", "track", "wbr"})
    SKIP = frozenset({"script", "style", "template", "noscript", "svg", "nav",
                      "footer", "header", "aside"})

    def __init__(self, *, nist_article_container: bool = False):
        super().__init__(convert_charrefs=True)
        # NIST SP 800-63B uses a unique <div class="container"> body
        # instead of <main>/<article>. Apply only to this exact approved URL.
        self.nist_article_container = nist_article_container
        self.depth = 0
        self.main_depth = None
        self.ignore_depth = None
        self.fragments = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.depth += 1
        attr = dict(attrs)
        if self.ignore_depth is None and tag in self.SKIP:
            self.ignore_depth = self.depth
        if self.main_depth is None and self.ignore_depth is None:
            css = (attr.get("class") or "").split()
            if (tag in ("main", "article") or attr.get("role") == "main"
                    or attr.get("id") == "question"
                    or (tag == "div" and "body" in css)
                    or (self.nist_article_container and tag == "div"
                        and "container" in css)):
                self.main_depth = self.depth
        if tag in self.BLOCKS and self.main_depth is not None and self.ignore_depth is None:
            self.fragments.append("\n")

    def handle_endtag(self, tag):
        if self.main_depth is not None and self.ignore_depth is None and tag in self.BLOCKS:
            self.fragments.append("\n")
        if self.ignore_depth is not None and self.depth <= self.ignore_depth:
            self.ignore_depth = None
        if self.main_depth is not None and self.depth <= self.main_depth:
            self.main_depth = None
        if tag not in self.VOID:
            self.depth = max(self.depth - 1, 0)

    def handle_data(self, data):
        if self.main_depth is not None and self.ignore_depth is None:
            self.fragments.append(data)

    def content(self) -> str:
        return "\n".join(
            clean(row, 650) for row in "".join(self.fragments).splitlines()
            if clean(row, 650)
        )


# Model-specific price tables are easily lost on long pages when a generic
# keyword scorer returns six unrelated lines from other models.
_PRICING_MODEL_SECTIONS = (
    ("gemini-3.5-flash-lite", re.compile(r"^\s*#{0,3}\s*Gemini\s+3\.5\s+Flash[- ]Lite\b", re.I)),
    ("gemini-4-carbon", re.compile(r"^\s*#{0,3}\s*Gemini\s+4\s+Carbon\b", re.I)),
)


def _targeted_pricing_excerpt(text: str, question: str) -> str:
    """Extract named models' Standard pricing rows, never another model's rows.

    Target detection is strict: it only uses a section actually present in the
    fetched document. No static/free-tier claims are injected by this method.
    """
    q = question.casefold()
    requested = [(name, heading) for name, heading in _PRICING_MODEL_SECTIONS
                 if name in q or name.replace("-", " ") in q]
    if not requested:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    found = []
    for model, heading in requested:
        indices = [i for i, line in enumerate(lines) if heading.match(line)]
        if not indices:
            continue
        start = indices[0]
        end = len(lines)
        for i in range(start + 1, len(lines)):
            line = lines[i]
            # Stop before Batch/Flex/Priority, or a new model heading.
            if (re.match(r"^#{1,3}\s*(Batch|Flex|Priority)\b", line, flags=re.I)
                    or re.match(r"^#{1,2}\s*Gemini\s+\d", line, flags=re.I)
                    or re.match(r"^Gemini\s+\d+(?:[.]\d+)?\s+(?!API\b)", line, flags=re.I)):
                end = i
                break
        section = "\n".join(lines[start:end])
        # Require the actual model identifier and a pricing row nearby.
        if (heading.match(lines[start])
                and re.search(r"input price|output price|free tier", section, flags=re.I)):
            found.append(section)
    if not found:
        return ""
    per_model = MAX_EXCERPT if len(found) == 1 else MAX_EXCERPT // len(found)
    return "\n\n".join(section[:per_model] for section in found)[:MAX_EXCERPT].strip()


def _privacy_terms_excerpt(text: str) -> str:
    """Extract genuine unpaid/paid policy paragraphs from official documentation.

    Google's official .md representation wraps sentences across lines and uses
    heading IDs, e.g. ## Unpaid Services {% id="unpaid-services" %}.
    Rejoin soft wraps before matching. Do not invent missing clauses.
    """
    if "Gemini API Additional Terms of Service" not in text:
        return ""
    # Only these two H2s define the relevant sections. Optional escaped hashes
    # and link/anchor annotations are accepted; arbitrary matches are not.
    heading = lambda name: r"(?m)^\\?#{2}\s+" + re.escape(name) + r"\b[^\n]*$"
    unpaid_match = re.search(heading("Unpaid Services"), text, re.I)
    paid_match = re.search(heading("Paid Services"), text, re.I)
    if not unpaid_match or not paid_match or paid_match.start() <= unpaid_match.end():
        return ""
    unpaid_part = text[unpaid_match.end():paid_match.start()]
    next_h2 = re.search(r"(?m)^\\?#{2}\s+\w", text[paid_match.end():])
    paid_part = text[paid_match.end():paid_match.end() + next_h2.start()] if next_h2 else text[paid_match.end():]

    def paragraphs(raw: str) -> list[str]:
        # Keep paragraph boundaries but join document soft line wraps.
        cleaned = re.sub(r"\r\n?", "\n", raw)
        cleaned = re.sub(r"(?m)^\\?#{3,6}\s+[^\n]*\n?", "\n\n", cleaned)
        return [
            clean(" ".join(p.splitlines()), 850)
            for p in re.split(r"\n\s*\n", cleaned)
            if p.strip()
        ]

    unpaid = paragraphs(unpaid_part)
    paid = paragraphs(paid_part)
    def find_one(paras, *tokens):
        return next((p for p in paras if all(x in p.casefold() for x in tokens)), "")
    unpaid_use = find_one(unpaid, "when you use unpaid services", "google uses")
    reviewer = find_one(unpaid, "human reviewers may", "api input and output")
    paid_use = find_one(paid, "when you use paid services", "doesn't use your prompts")
    if not paid_use:
        paid_use = find_one(paid, "when you use paid services", "does not use your prompts")
    if not unpaid_use or not reviewer or not paid_use:
        return ""
    paid_qualifier = find_one(paid, "your access to gemini api", "active billing account")
    paid_logging = find_one(paid, "for paid services", "logs prompts")
    # Per-paragraph clipping is bounded; sections stay distinct. All clauses
    # appear in the actually downloaded official document.
    result = (
        "Unpaid Services\n" + unpaid_use[:680] + "\n" + reviewer[:680] +
        "\n\nPaid Services\n" + paid_use[:680] +
        ("\n" + paid_qualifier[:240] if paid_qualifier else "") +
        ("\n" + paid_logging[:250] if paid_logging else "")
    )
    return result[:MAX_EXCERPT]


def select_relevant_excerpt(text: str, query: str) -> str:
    """Prefer exact named pricing sections, then generic topical excerpts."""
    targeted = _targeted_pricing_excerpt(text, query)
    if targeted:
        return targeted
    terms = set(search_query_terms(query))
    parts = [p.strip() for p in text.splitlines() if len(p.strip()) >= 22]
    if not parts:
        return ""
    ranked = []
    for i, part in enumerate(parts):
        words = set(re.findall(r"[a-z0-9]+", part.lower()))
        overlap = len(terms & words)
        ranked.append((overlap, i, part))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if terms and ranked[0][0] == 0:
        return ""
    selected = sorted(ranked[:6], key=lambda x: x[1])
    excerpt = "\n".join(p for _, _, p in selected)
    return excerpt[:MAX_EXCERPT].strip()


async def fetch_page_excerpt(
    url: str, question: str, *, client: Optional[httpx.AsyncClient] = None
) -> str:
    """Bounded HTTPS GET; permit only exact allowlisted redirects and document media."""
    if not is_approved_document_url(url):
        return ""

    async def execute(session: httpx.AsyncClient) -> str:
        current = url
        try:
            # One checked same-host redirect at most. No client-wide redirects.
            # Never follow to login, arbitrary Google Search, internal hosts, etc.
            for attempt in range(2):
                async with session.stream(
                    "GET", current, follow_redirects=False,
                    headers={"Accept": "text/html"}, timeout=FETCH_TIMEOUT,
                ) as resp:
                    if resp.status_code in (301, 302, 307, 308) and attempt == 0:
                        location = resp.headers.get("location", "")
                        target = urljoin(current, location)
                        if (not location or not is_approved_document_url(target)
                                or urlsplit(target).hostname != urlsplit(current).hostname
                                or target == current):
                            log.info("Document redirected to unapproved destination host=%s",
                                     urlsplit(current).hostname)
                            return ""
                        current = target
                        continue
                    if resp.status_code != 200:
                        log.info("Document excerpt unavailable host=%s status=%s",
                                 urlsplit(current).hostname, resp.status_code)
                        return ""
                    mime = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    host = urlsplit(current).hostname or ""
                    is_studio = host == "aistudio.google.com" and is_approved_document_url(current)
                    html_mime = mime in ("text/html", "application/xhtml+xml")
                    text_mime = mime in ("text/plain", "text/markdown")
                    # Aistudio documentation may be returned as Markdown or plain text
                    # rather than HTML. Only allow these on exact approved URLs.
                    if not html_mime and not (is_studio and text_mime):
                        log.info("Document excerpt unsupported media host=%s type=%s",
                                 host, mime[:60] or "missing")
                        return ""
                    length = resp.headers.get("content-length", "")
                    if length.isdecimal() and int(length) > MAX_PAGE_BYTES:
                        return ""
                    buf = bytearray()
                    async for piece in resp.aiter_bytes():
                        buf.extend(piece)
                        if len(buf) > MAX_PAGE_BYTES:
                            return ""
                raw = buf.decode("utf-8", errors="replace")
                # HTML pages must have an actual article/main region. A JS shell
                # or generic login page must not count as document evidence.
                if html_mime:
                    parser = MainText(
                        nist_article_container=(current == IDENTITY_DOCS[0][0])
                    )
                    parser.feed(raw)
                    page_text = parser.content()
                else:
                    # Reject generic anti-bot/error messages and content without
                    # a specific AI Studio/Gemini API subject mention.
                    if len(raw.strip()) < 100 or not any(
                        term in raw.casefold() for term in
                        ("google ai studio", "gemini api", "gemini model", "free tier")
                    ):
                        log.info("Document excerpt rejected low-content text host=%s", host)
                        return ""
                    page_text = "\n".join(
                        clean(line, 650) for line in raw.splitlines()
                        if clean(line, 650)
                    )
                if current == GOOGLE_DOCS["api_terms"][0]:
                    excerpt = _privacy_terms_excerpt(page_text)
                else:
                    excerpt = select_relevant_excerpt(page_text, question)
                if not excerpt:
                    log.info("Document excerpt no topical text host=%s",
                             urlsplit(current).hostname)
                return excerpt
            return ""
        except (httpx.HTTPError, UnicodeError, ValueError):
            log.info("Document excerpt request failed host=%s", urlsplit(current).hostname)
            return ""

    if client is not None:
        return await execute(client)
    async with httpx.AsyncClient(follow_redirects=False) as session:
        return await execute(session)


def recommended_official_docs(question: str) -> list[dict]:
    """Select relevant allowlisted docs by named topic, not generic phrasing.

    A standalone aiolimiter query MUST select the aiolimiter docs, without
    requiring Python/asyncio to be mentioned explicitly.
    """
    terms = set(search_query_terms(question))
    refs = []
    if "python" in terms or "asyncio" in terms or "semaphore" in terms:
        if "semaphore" in terms or "synchronization" in terms:
            url, title = OFFICIAL_SEMAPHORE, "asyncio synchronization primitives (Python docs)"
        elif "asyncio" in terms:
            url, title = OFFICIAL_ASYNCIO, "asyncio (Python official docs)"
        else:
            url, title = None, None
        if url:
            refs.append({"provider": "Official Python documentation", "title": title,
                         "url": url, "snippet": "", "published": ""})
    if "aiolimiter" in terms or ("asyncio" in terms and (
        "rate" in terms or "limiting" in terms or "limiter" in terms
        or ("leaky" in terms and "bucket" in terms)
    )):
        refs.append({"provider": "aiolimiter official documentation",
                     "title": "aiolimiter asyncio rate limiting documentation",
                     "url": AIOLIMITER_DOCS, "snippet": "", "published": ""})
    return refs


async def enrich_with_read_excerpts(
    question: str, sources: list[dict], *,
    client: Optional[httpx.AsyncClient] = None
) -> list[dict]:
    """Read at most three source pages normally, or eight curated Google pages.

    An official reference is only added if its page actually returns a relevant
    excerpt. Never label unvisited links as read.
    """
    approved_candidates = []
    seen = set()
    official_google = google_official_docs(question)
    identity_docs = identity_official_docs(question)
    capacity = MAX_GOOGLE_FETCHES if len(official_google) >= 3 else MAX_FETCHES
    for ref in recommended_official_docs(question) + official_google + identity_docs + sources:
        url = ref.get("url", "")
        if url in seen or not is_approved_document_url(url):
            continue
        seen.add(url)
        approved_candidates.append(ref)
        if len(approved_candidates) >= capacity:
            break

    async def load(ref):
        focus = ref.get("focus", question)
        if (ref["url"] == GOOGLE_DOCS["studio_free"][0]
                and any(model in question.casefold() for model, _ in _PRICING_MODEL_SECTIONS)):
            # The specific model ID must reach section extraction; the fixed
            # generic "pricing" focus previously hid it completely.
            focus = question
        excerpt = await fetch_page_excerpt(ref["url"], focus, client=client)
        if not excerpt:
            return None
        return {**ref, "excerpt": excerpt, "read_status": "verified_page_excerpt"}

    loaded = await asyncio.gather(*(load(x) for x in approved_candidates))
    validated = [item for item in loaded if item]
    read_urls = {item["url"] for item in validated}
    remaining = [{**s, "read_status": "search_metadata_only"} for s in sources
                 if s.get("url") not in read_urls]
    # Official content first, community/search metadata second.
    return (validated + remaining)[:8]
