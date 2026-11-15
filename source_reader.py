"""Strictly bounded read-only source excerpts for /deepdive.

Only pre-approved public documentation or Stack Overflow question pages can
be fetched. Arbitrary result URLs are never requested. Redirects are disabled.
Returned text is UNTRUSTED evidence, not a source of instructions.
"""
import asyncio
import html
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit
from typing import Optional

import httpx
from free_research import clean, search_query_terms

log = logging.getLogger("source_reader")
MAX_PAGE_BYTES = 450_000
MAX_EXCERPT = 2400
MAX_FETCHES = 3
FETCH_TIMEOUT = 7.0

OFFICIAL_ASYNCIO = "https://docs.python.org/3/library/asyncio.html"
OFFICIAL_SEMAPHORE = "https://docs.python.org/3/library/asyncio-sync.html"
AIOLIMITER_DOCS = "https://aiolimiter.readthedocs.io/en/stable/"


def is_approved_document_url(url: str) -> bool:
    """Avoid SSRF by fetching only exact hosts and tightly scoped HTTPS paths."""
    if not isinstance(url, str) or len(url) > 700 or any(c in url for c in "\r\n\t<>"):
        return False
    try:
        p = urlsplit(url)
        if p.scheme != "https" or p.port not in (None, 443) or p.username or p.password:
            return False
        host, path = p.hostname or "", p.path
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

    def __init__(self):
        super().__init__(convert_charrefs=True)
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
                    or (tag == "div" and "body" in css)):
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


def select_relevant_excerpt(text: str, query: str) -> str:
    """Take the most relevant page sections, not just the navigation / first lines."""
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
    """Bounded HTTPS GET; never follow redirects or download non-HTML."""
    if not is_approved_document_url(url):
        return ""

    async def execute(session: httpx.AsyncClient) -> str:
        try:
            async with session.stream(
                "GET", url, follow_redirects=False,
                headers={"Accept": "text/html"}, timeout=FETCH_TIMEOUT,
            ) as resp:
                if resp.status_code != 200:
                    log.info("Document excerpt unavailable host=%s status=%s",
                             urlsplit(url).hostname, resp.status_code)
                    return ""
                mime = resp.headers.get("content-type", "").lower()
                if "text/html" not in mime:
                    return ""
                length = resp.headers.get("content-length", "")
                if length.isdecimal() and int(length) > MAX_PAGE_BYTES:
                    return ""
                buf = bytearray()
                async for piece in resp.aiter_bytes():
                    buf.extend(piece)
                    if len(buf) > MAX_PAGE_BYTES:
                        return ""
            parser = MainText()
            parser.feed(buf.decode("utf-8", errors="replace"))
            return select_relevant_excerpt(parser.content(), question)
        except (httpx.HTTPError, UnicodeError, ValueError):
            log.info("Document excerpt request failed host=%s", urlsplit(url).hostname)
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
    """Read at most three pre-approved source pages; all other hits remain metadata.

    An official reference is only added if its page actually returns a relevant
    excerpt. Never label unvisited links as read.
    """
    approved_candidates = []
    seen = set()
    for ref in recommended_official_docs(question) + sources:
        url = ref.get("url", "")
        if url in seen or not is_approved_document_url(url):
            continue
        seen.add(url)
        approved_candidates.append(ref)
        if len(approved_candidates) >= MAX_FETCHES:
            break

    async def load(ref):
        excerpt = await fetch_page_excerpt(ref["url"], question, client=client)
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
