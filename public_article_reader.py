"""Bounded read-only retrieval of public search-result articles.

Only URLs returned by the public search index are candidates. The transport
resolves and validates every DNS answer, then connects to one approved public
IP while preserving the original HTTPS hostname for SNI/certificate checks.
No proxy, redirects, browser actions, authenticated content or binary files.
"""
import asyncio
import ipaddress
import re
import socket
import ssl
from urllib.parse import urlsplit

import httpcore
import httpx
from httpcore._backends.anyio import AnyIOBackend

BLOCKED_SUFFIXES = (".local", ".localhost", ".localdomain", ".internal", ".home", ".lan", ".test", ".invalid", ".onion")
BLOCKED_PATHS = ("/admin", "/login", "/signin", "/account", "/oauth",
                 "/api/", "/wp-json", "/.git", "/.env", "/config",
                 "/download", "/logout", "/delete", "/graphql")
BLOCKED_EXTENSIONS = (".pdf", ".zip", ".tar", ".gz", ".exe", ".bin",
                      ".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif",
                      ".mp4", ".mp3", ".xml", ".json", ".csv", ".xlsx",
                      ".docx", ".pptx", ".js", ".css")
# Bounded: official documentation HTML can exceed 400 kB while still being simple text.
# Maximum of three reads means at most 1.5 MB of retained HTML per request.
MAX_GENERAL_BYTES = 500_000


def safe_public_article_url(url: str) -> bool:
    """Syntactic filter; DNS/public-IP enforcement happens at connect time."""
    if not isinstance(url, str) or len(url) > 650 or any(ord(c) < 33 or c in '<>\\' for c in url):
        return False
    try:
        p = urlsplit(url)
        host = p.hostname or ""
        if (p.scheme != "https" or not host or p.port not in (None, 443)
                or p.username or p.password or p.query or p.fragment
                or not re.fullmatch(r"[a-zA-Z0-9.-]{4,250}", host)
                or ".." in host or host.startswith("-")
                or host.endswith("-") or "." not in host
                or host.casefold().endswith(BLOCKED_SUFFIXES)):
            return False
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            pass
        path = p.path or "/"
        low = path.casefold()
        if ("%" in low or "//" in path or "/.." in path or
                any(segment in low for segment in BLOCKED_PATHS) or
                low.endswith(BLOCKED_EXTENSIONS)):
            return False
        return bool(re.fullmatch(r"/[a-zA-Z0-9_./~-]{0,300}", path))
    except (ValueError, UnicodeError):
        return False


async def resolve_public_ip(host: str, port: int = 443) -> str:
    """Fail closed for private/mixed DNS, and bound DNS resolution time.

    Only the selected validated IP is passed to the transport's TCP dialer.
    This prevents DNS rebinding between validation and connection.
    """
    if (port != 443 or not re.fullmatch(r"[a-zA-Z0-9.-]{4,250}", host)
            or not safe_public_article_url("https://" + host + "/article")):
        raise OSError("Blocked destination")
    try:
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            ), timeout=2.5
        )
        ips = [ipaddress.ip_address(row[4][0]) for row in addresses]
    except (asyncio.TimeoutError, ValueError, OSError) as exc:
        raise OSError("DNS lookup failed or timed out") from exc
    if not ips or any(not ip.is_global for ip in ips):
        raise OSError("Non-public destination")
    # Prefer IPv4 where available, but use a validated global IPv6 otherwise.
    return str(next((ip for ip in ips if ip.version == 4), ips[0]))


class PublicIPBackend(AnyIOBackend):
    """Prevent DNS rebinding: transport dials the validated IP, not hostname."""
    async def connect_tcp(self, host, port, timeout=None, local_address=None,
                          socket_options=None):
        addr = await resolve_public_ip(host, port)
        # httpcore still performs TLS with the ORIGINAL hostname as SNI and
        # verifies its certificate; only the TCP destination is pinned.
        return await super().connect_tcp(
            addr, port, timeout=timeout, local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise OSError("Unix sockets not permitted")


class PublicArticleTransport(httpx.AsyncHTTPTransport):
    """Fixed zero-retry direct HTTPS transport; environment proxies disabled."""
    def __init__(self):
        super().__init__(trust_env=False, retries=0,
                         limits=httpx.Limits(max_connections=3,
                                             max_keepalive_connections=0))
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            network_backend=PublicIPBackend(),
            max_connections=3, max_keepalive_connections=0, retries=0,
        )


async def read_public_html(url: str, *,
                           client: httpx.AsyncClient | None = None) -> str:
    """Fetch at most one public HTML page with no redirects or retries."""
    if not safe_public_article_url(url):
        return ""

    async def request(session):
        try:
            async with session.stream(
                "GET", url, follow_redirects=False,
                headers={"Accept": "text/html,application/xhtml+xml"},
                timeout=httpx.Timeout(7.0, connect=4.0),
            ) as resp:
                if resp.status_code != 200:
                    return ""
                mime = resp.headers.get("content-type", "").split(";", 1)[0].lower().strip()
                if mime not in ("text/html", "application/xhtml+xml"):
                    return ""
                size = resp.headers.get("content-length", "")
                if size.isdecimal() and int(size) > MAX_GENERAL_BYTES:
                    return ""
                buf = bytearray()
                async for part in resp.aiter_bytes():
                    buf.extend(part)
                    if len(buf) > MAX_GENERAL_BYTES:
                        return ""
                return buf.decode("utf-8", errors="replace")
        except (httpx.HTTPError, OSError, ValueError, UnicodeError):
            return ""

    if client is not None:
        return await request(client)
    async with httpx.AsyncClient(transport=PublicArticleTransport(),
                                  follow_redirects=False, trust_env=False) as session:
        return await request(session)
