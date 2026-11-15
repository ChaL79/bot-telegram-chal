"""Real HTTPS/TLS smoke tests. Only run in an isolated PR Actions job.

Never executed by the normal unit-test suite. No API keys, no credentials,
no deployment, and no Telegram messages.
"""
import os
import unittest

from public_article_reader import read_public_html


@unittest.skipUnless(os.environ.get("RUN_NETWORK_SMOKE") == "1",
                     "Explicit network-smoke flag required")
class LivePublicReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_pinned_https_uses_real_dns_tls_and_certificate_validation(self):
        # Small standards-compliant public HTML page; reliable baseline that
        # should not violate the reader's 350 KB size bound.
        page = await read_public_html("https://example.com/")
        self.assertIn("Example Domain", page,
                      "Pinned-IP HTTPS/TLS failed to fetch a small public page")

    async def test_wcag_fallback_urls_real_network_diagnostic(self):
        # Exactly the primary-source URLs used by the opt-in WCAG fallback.
        # Fail independently by source only when tests are converted to strict.
        from source_reader import (
            ACCESSIBILITY_OFFICIAL_DOCS, MainText, select_relevant_excerpt,
        )
        question = "Bandingkan WCAG 2.2 daripada W3C, MDN dan GOV.UK."
        available = []
        for org, (url, _title, host) in ACCESSIBILITY_OFFICIAL_DOCS.items():
            html = await read_public_html(url)
            excerpt = ""
            if html:
                parser = MainText()
                parser.feed(html)
                excerpt = select_relevant_excerpt(parser.content(), question)
            print(f"WCAG LIVE FALLBACK org={org} host={host} "
                  f"html_chars={len(html)} excerpt_chars={len(excerpt)}", flush=True)
            if excerpt and len(excerpt) >= 65:
                available.append(org)
        self.assertTrue(available, "No real WCAG fallback extracts readable evidence")

    async def test_wcag_public_institution_attempts(self):
        # At least one real organization must return usable HTML. Some pages
        # can legitimately redirect, block CI or exceed the size bound.
        targets = {
            "W3C": "https://www.w3.org/WAI/standards-guidelines/wcag/docs/",
            "MDN": "https://developer.mozilla.org/en-US/docs/Web/Accessibility",
            "GOV.UK": "https://www.gov.uk/service-manual/helping-people-to-use-your-service/testing-for-accessibility",
        }
        successes = []
        for org, url in targets.items():
            html = await read_public_html(url)
            if html and len(html) > 100:
                successes.append(org)
        self.assertTrue(successes,
                        "All WCAG organization pages unavailable to secure reader "
                        "(redirect, block, oversized response or TLS failure)")


if __name__ == "__main__":
    unittest.main()
