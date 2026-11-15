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

log = logging.getLogger("free_research")
MAX_BYTES = 550_000
MAX_SOURCES = 8
WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChaL79Bot/1.4; personal research)",
    "Accept": "application/rss+xml,application/xml,application/json,text/xml;q=0.8,*/*;q=0.1",
}


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


async def public_source_search(query: str, *, client: httpx.AsyncClient | None = None) -> list[dict]:
    """Search Bing RSS, Wikipedia API and Google News RSS concurrently.

    Uses only public, read-only endpoints. Fails gracefully if hosts block Render.
    No private network fetches and no additional paid API keys.
    """
    query = clean(query, 210)
    if not query:
        return []

    async def search(session: httpx.AsyncClient) -> list[dict]:
        async def get_bing():
            resp = await session.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss", "mkt": "en-US"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                return []
            return parse_rss(resp.content, "Bing web-search RSS", 5)

        async def get_wikipedia():
            resp = await session.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query,
                        "srlimit": "3", "format": "json"},
                timeout=7.0,
            )
            if resp.status_code != 200 or len(resp.content) > MAX_BYTES:
                return []
            return parse_wikipedia(resp.json(), 3)

        async def get_news():
            resp = await session.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                timeout=7.0,
            )
            if resp.status_code != 200:
                return []
            return parse_rss(resp.content, "Google News RSS (headline)", 3)

        results = await asyncio.gather(
            get_bing(), get_wikipedia(), get_news(), return_exceptions=True
        )
        sources, urls = [], set()
        for result in results:
            if isinstance(result, BaseException):
                log.warning("Public source feed unavailable: %s", type(result).__name__)
                continue
            for item in result:
                if item["url"] in urls:
                    continue
                urls.add(item["url"])
                sources.append(item)
                if len(sources) >= MAX_SOURCES:
                    return sources
        return sources

    if client is not None:
        return await search(client)
    async with httpx.AsyncClient(headers=WEB_HEADERS, follow_redirects=True) as new_client:
        return await search(new_client)


def evidence_text(sources: list[dict]) -> str:
    """A clear data boundary. The text is search metadata and must not be obeyed."""
    chunks = []
    for i, source in enumerate(sources[:MAX_SOURCES], start=1):
        parts = [
            f"[{i}] Provider: {clean(source['provider'], 50)}",
            "Title: " + clean(source["title"], 115),
            "URL: " + source["url"],
            "Snippet/headline only: " + clean(source.get("snippet", ""), 300),
        ]
        if source.get("published"):
            parts.append("Feed date: " + clean(source["published"], 55))
        chunks.append("\n".join(parts))
    return "\n\n".join(chunks)


def references_text(sources: list[dict]) -> str:
    return "\n\nPautan hasil carian (metadata, bukan semakan artikel penuh):\n" + "\n".join(
        f"[{i}] {clean(source['title'], 95)}\n{source['url']}"
        for i, source in enumerate(sources[:MAX_SOURCES], start=1)
    )
