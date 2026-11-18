"""Public official alternate-origin live probe without authentication.

For diagnoses only. Every candidate is a reviewed literal public HTML URL
validated by the same pinned-DNS reader, never pulled from arbitrary user input.
No proxy, browser or redirect following. Output contains no article body.
"""
import asyncio
import json
from source_reader import MainText, BoundedBodyText
from public_article_reader import read_public_html

CANDIDATES = {
    "IPCC_REPORT_HOST": "https://report.ipcc.ch/ar6syr/index.html",
    "IPCC_AR6_SPM": "https://www.ipcc.ch/report/ar6/syr/summary-for-policymakers/",
    "IPCC_DDC": "https://ipcc-data.climate.columbia.edu/content/ipcc-ar6-syr-lr-figure-34",
    "ASEAN_ALTERNATE_PORTAL": "https://afamt.asean.org/about-asean/",
    "ASEAN_NEW_PORTAL": "https://asean.org/about-us/",
    "THAI_MFA_ASEAN": "https://asean.mfa.go.th/en/page/62206-the-birth-of-asean",
    "SINGAPORE_MFA_ASEAN": "https://www.mfa.gov.sg/about-mfa/foreign-policy/international-organisations/association-of-southeast-asian-nations-asean/",
    "PHILIPPINES_ORIGINAL_DECLARATION": "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/36/10681",
    "INDONESIA_MFA_FOUNDING": "https://kemlu.go.id/kebijakan/asean/sejarah-dan-latar-pembentukan-asean",
    "MALAYSIA_MFA_ASEAN": "https://www.kln.gov.my/web/chl_santiago/post/celebrating",
}

async def one(label, url):
    try:
        html = await asyncio.wait_for(read_public_html(url), timeout=10)
        extracted = ""
        if html:
            parser = MainText()
            parser.feed(html)
            extracted = parser.content()
            if not extracted:
                parser = BoundedBodyText()
                parser.feed(html)
                extracted = parser.content()
        lower = extracted.casefold()
        anchors = (
            ["ipcc", "sea level"] if label.startswith("IPCC") else
            ["asean", "1967"]
        )
        return {
            "case": label, "html": len(html), "text": len(extracted),
            "anchors": [key for key in anchors if key in lower],
            "readable": bool(extracted and all(key in lower for key in anchors)),
        }
    except (asyncio.TimeoutError, ValueError):
        return {"case": label, "readable": False, "reason": "timeout_or_invalid"}

async def main():
    for label, url in CANDIDATES.items():
        outcome = await one(label, url)
        print("OFFICIAL_ORIGIN_PROBE " + json.dumps(outcome), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
