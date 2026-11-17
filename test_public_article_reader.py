import unittest
import httpx
from unittest.mock import AsyncMock, patch

from public_article_reader import (
    safe_public_article_url, resolve_public_ip, read_public_html, PublicIPBackend, MAX_GENERAL_BYTES,
)
from source_reader import enrich_with_read_excerpts


class PublicReaderSecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_only_public_https_article_shapes(self):
        valid = [
            "https://www.w3.org/TR/WCAG22/",
            "https://developer.mozilla.org/en-US/docs/Web/Accessibility/Guides/Understanding_WCAG/Keyboard",
            "https://www.gov.uk/service-manual/helping-people-to-use-your-service/testing-for-accessibility",
        ]
        for url in valid:
            self.assertTrue(safe_public_article_url(url), url)
        unsafe = [
            "http://www.w3.org/TR/WCAG22/",
            "https://127.0.0.1/internal",
            "https://169.254.169.254/latest/meta-data",
            "https://[::1]/admin",
            "https://localhost/private",
            "https://localhost.localdomain/private",
            "https://internal.local/api/metadata",
            "https://evil.test/abc",
            "https://w3.org.evil.test/TR/WCAG22/",
            "https://user:secret@www.w3.org/TR/WCAG22/",
            "https://www.w3.org:8080/TR/WCAG22/",
            "https://www.w3.org/TR/WCAG22/?token=abc",
            "https://www.w3.org/TR/WCAG22/#fragment",
            "https://www.w3.org/%2e%2e/internal",
            "https://www.w3.org/.git/config",
            "https://www.w3.org/admin/users",
            "https://www.w3.org/TR/WCAG22/report.pdf",
            "https://www.w3.org//hidden",
            "https://www.w3.org/..//hidden",
            "https://www.w3.org/a b",
        ]
        for url in unsafe:
            self.assertFalse(safe_public_article_url(url), url)

    async def test_dns_fail_closed_when_private_or_mixed(self):
        async def mocked_getaddrinfo(self_loop, host, port, **kwargs):
            return [
                (2, 1, 6, "", ("93.184.215.14", port)),
                (2, 1, 6, "", ("127.0.0.1", port)),
            ]
        with patch("asyncio.BaseEventLoop.getaddrinfo", new=mocked_getaddrinfo):
            with self.assertRaises(OSError):
                await resolve_public_ip("example.org")
        async def only_public(self_loop, host, port, **kwargs):
            return [(2, 1, 6, "", ("93.184.215.14", port))]
        with patch("asyncio.BaseEventLoop.getaddrinfo", new=only_public):
            self.assertEqual(await resolve_public_ip("example.org"),
                             "93.184.215.14")

    async def test_dns_resolution_timeout_fails_closed(self):
        import asyncio
        async def hung_getaddrinfo(self_loop, host, port, **kwargs):
            raise asyncio.TimeoutError()
        with patch("asyncio.BaseEventLoop.getaddrinfo", new=hung_getaddrinfo):
            with self.assertRaises(OSError):
                await resolve_public_ip("www.w3.org")

    async def test_transport_dials_pinned_ip_and_preserves_tls_hostname(self):
        from httpcore._backends.anyio import AnyIOBackend
        calls = []
        async def checked(host, port):
            self.assertEqual((host, port), ("www.w3.org", 443))
            return "93.184.215.14"
        async def dial(self_obj, host, port, **kwargs):
            calls.append((host, port))
            return object()
        with patch("public_article_reader.resolve_public_ip", new=checked):
            with patch.object(AnyIOBackend, "connect_tcp", new=dial):
                await PublicIPBackend().connect_tcp("www.w3.org", 443)
        self.assertEqual(calls, [("93.184.215.14", 443)])

    async def test_public_ipv6_and_tls_failure_handled_safely(self):
        import ssl
        import socket
        async def ipv6_only(self_loop, host, port, **kwargs):
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "",
                     ("2606:4700:4700::1111", port, 0, 0))]
        with patch("asyncio.BaseEventLoop.getaddrinfo", new=ipv6_only):
            self.assertEqual(await resolve_public_ip("example.org"),
                             "2606:4700:4700::1111")
        url = "https://www.w3.org/TR/WCAG22/"
        calls = []
        def tls_failure(req):
            calls.append(str(req.url))
            raise httpx.ConnectError("certificate verify failed", request=req)
        async with httpx.AsyncClient(transport=httpx.MockTransport(tls_failure)) as cli:
            self.assertEqual(await read_public_html(url, client=cli), "")
        self.assertEqual(calls, [url])

    async def test_no_redirect_no_binary_no_excessive_content(self):
        url = "https://www.w3.org/TR/WCAG22/"
        for response in [
            httpx.Response(302, headers={"location":"http://127.0.0.1/private"}),
            httpx.Response(200, headers={"content-type":"application/pdf"},
                           content=b"%PDF"),
            httpx.Response(200, headers={"content-type":"text/html",
                                         "content-length":"999999"}),
            httpx.Response(200, headers={"content-type":"text/html"},
                           content=b"a"*(MAX_GENERAL_BYTES + 1)),
        ]:
            requests = []
            def mock(req):
                requests.append(str(req.url))
                return response
            async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
                self.assertEqual(await read_public_html(url, client=cli), "")
            self.assertEqual(requests, [url])

    async def test_bounded_document_above_350kb_is_allowed(self):
        url = "https://semiconductor.samsung.com/consumer-storage/support/faqs/internalssd-product-information/"
        content = b"<html><main><p>Samsung NVMe SSD SATA compatibility documentation.</p></main>" + b" " * 430000 + b"</html>"
        self.assertLessEqual(len(content), MAX_GENERAL_BYTES)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, headers={"content-type": "text/html"}, content=content)
        )) as cli:
            self.assertEqual(await read_public_html(url, client=cli), content.decode())

    async def test_general_reader_can_be_disabled_explicitly(self):
        from unittest.mock import patch
        sources = [{"title": "WCAG 2.2 keyboard accessibility",
                    "url": "https://www.w3.org/TR/WCAG22/",
                    "provider": "Bing RSS", "snippet": "Keyboard focus",
                    "published": ""}]
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: self.fail("Reader must not fetch with flag disabled")
        )) as cli:
            with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER":"0"}, clear=True):
                result = await enrich_with_read_excerpts(
                    "WCAG 2.2 keyboard focus", sources, client=cli)
        self.assertEqual(result[0]["read_status"], "search_metadata_only")

    async def test_search_discovered_multi_domain_excerpts_without_allowlists(self):
        sources = [
            {"title":"WCAG 2.2 Focus Not Obscured","url":"https://www.w3.org/TR/WCAG22/",
             "provider":"Bing RSS", "snippet":"Focus visibility keyboard","published":""},
            {"title":"MDN keyboard accessibility","url":"https://developer.mozilla.org/en-US/docs/Web/Accessibility/Keyboard",
             "provider":"Bing RSS", "snippet":"Keyboard focus testing","published":""},
        ]
        requested = []
        def mock(req):
            requested.append(str(req.url))
            return httpx.Response(200, headers={"content-type":"text/html"},
                content=b"<html><main><h1>WCAG 2.2 Keyboard Accessibility</h1>"
                b"<p>Keyboard focus visibility and keyboard operation are essential."
                b" Focus Not Obscured helps people navigate pages.</p></main></html>")
        question = "Compare WCAG 2.2 keyboard accessibility focus visibility testing"
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "1"}):
                results = await enrich_with_read_excerpts(question, sources, client=client)
        self.assertEqual(len(requested), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(x["read_status"]=="verified_page_excerpt" for x in results))
        self.assertTrue(all("Keyboard" in x["excerpt"] for x in results))

    async def test_independent_domain_priority(self):
        urls = [
            "https://www.w3.org/WAI/standards-guidelines/wcag/docs/",
            "https://www.w3.org/WAI/standards-guidelines/wcag/",
            "https://www.w3.org/TR/WCAG22/",
            "https://developer.mozilla.org/en-US/docs/Web/Accessibility/Keyboard",
            "https://www.gov.uk/service-manual/helping-people-to-use-your-service/testing-for-accessibility",
        ]
        sources = [{"title":"WCAG 2.2 keyboard and focus visibility",
                    "url":url, "provider":"Bing RSS", "snippet":"WCAG keyboard focus",
                    "published":""} for url in urls]
        visited = []
        def mock(req):
            visited.append(str(req.url))
            return httpx.Response(200, headers={"content-type":"text/html"},
                content=b"<html><main><h1>WCAG Keyboard Focus</h1>"
                        b"<p>WCAG keyboard focus visibility and testing"
                        b" support accessible interfaces.</p></main></html>")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "1"}):
                result = await enrich_with_read_excerpts(
                    "Compare WCAG keyboard focus visibility testing",
                    sources, client=cli)
        self.assertEqual(len(visited), 3)
        self.assertEqual([httpx.URL(u).host for u in visited],
                         ["www.w3.org","developer.mozilla.org","www.gov.uk"])
        self.assertEqual(sum(x["read_status"] == "verified_page_excerpt"
                             for x in result), 3)

    def test_curated_fallback_requires_explicit_wcag_institutions(self):
        from source_reader import accessibility_official_fallbacks, requested_wcag_hosts
        query = "Bandingkan WCAG 2.2 daripada W3C, MDN dan GOV.UK."
        hosts = requested_wcag_hosts(query)
        self.assertEqual(hosts, {"www.w3.org", "developer.mozilla.org", "www.gov.uk"})
        refs = accessibility_official_fallbacks(query, [])
        self.assertEqual({httpx.URL(x["url"]).host for x in refs}, hosts)
        self.assertEqual(accessibility_official_fallbacks("What is WCAG?", []), [])
        self.assertEqual(accessibility_official_fallbacks("W3C MDN GOV.UK overview", []), [])
        self.assertEqual(accessibility_official_fallbacks(
            query, [{"url": "https://developer.mozilla.org/en-US/docs/Web/Accessibility/Keyboard"}]
        ).__len__(), 2)

    async def test_wcag_missing_institutions_are_fetched_as_primary_evidence(self):
        question = "Bandingkan WCAG 2.2 daripada W3C, MDN dan GOV.UK."
        sources = [{"title": "WCAG 2.2 overview", "url": "https://www.w3.org/WAI/standards-guidelines/wcag/",
                    "provider": "Bing RSS", "snippet": "Accessibility WCAG", "published": ""}]
        visited = []
        def mock(req):
            visited.append(str(req.url))
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  content=b"<html><main><h1>WCAG 2.2 accessibility</h1>"
                                          b"<p>WCAG accessibility guidance for public services, developers,"
                                          b" and accessible web content.</p></main></html>")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "1"}):
                items = await enrich_with_read_excerpts(question, sources, client=cli)
        self.assertEqual({httpx.URL(url).host for url in visited},
                         {"www.w3.org", "developer.mozilla.org", "www.gov.uk"})
        self.assertEqual(len(visited), 3)
        self.assertEqual(sum(x["read_status"] == "verified_page_excerpt" for x in items), 3)

    async def test_failed_curated_fallback_never_claims_to_be_read(self):
        question = "WCAG 2.2 comparison MDN and GOV.UK"
        def mock(req):
            return httpx.Response(403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as cli:
            with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "1"}):
                items = await enrich_with_read_excerpts(question, [], client=cli)
        self.assertEqual(len(items), 2)
        self.assertTrue(all(x["read_status"] == "page_unavailable" for x in items))
        from free_research import references_text, evidence_text
        self.assertNotIn("Petikan halaman diambil", references_text(items))
        self.assertNotIn("FETCHED PAGE EXCERPT", evidence_text(items))

    async def test_disabled_flag_keeps_curated_fallback_off(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: self.fail("disabled reader must not fetch")
        )) as cli:
            with patch.dict("os.environ", {"ENABLE_GENERAL_PUBLIC_READER": "0"}):
                items = await enrich_with_read_excerpts(
                    "WCAG 2.2 comparison W3C MDN GOV.UK", [], client=cli)
        self.assertEqual(items, [])

    async def test_malicious_metadata_rejected_before_network(self):
        sources = [{"title":"evil","url":"https://127.0.0.1/api/token",
                    "provider":"Bing RSS", "snippet":"WCAG","published":""}]
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: self.fail("No request allowed")
        )) as client:
            items = await enrich_with_read_excerpts("WCAG keyboard", sources,client=client)
        self.assertEqual(items[0]["read_status"],"search_metadata_only")


if __name__ == "__main__":
    unittest.main()
