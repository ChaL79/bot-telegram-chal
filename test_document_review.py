"""Document review policy: accept owner-provided documents while not trusting
instructions embedded in them; don't invent a certificate of authenticity.
"""
import ast
import json
from pathlib import Path
import unittest
import httpx

from document_review import DOCUMENT_REVIEW_PROMPT
from media_tools import analyze_image, analyze_pdf


class DocumentReviewPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_policy_is_not_blanket_rejection(self):
        self.assertIn("jangan menolak", DOCUMENT_REVIEW_PROMPT)
        self.assertIn("Kesahihan rasmi", DOCUMENT_REVIEW_PROMPT)
        self.assertIn("belum dapat", DOCUMENT_REVIEW_PROMPT)
        self.assertIn("penerbit", DOCUMENT_REVIEW_PROMPT)
        self.assertIn("nombor rujukan", DOCUMENT_REVIEW_PROMPT)
        self.assertIn("JANGAN", DOCUMENT_REVIEW_PROMPT)
        self.assertIn("ringkasan", DOCUMENT_REVIEW_PROMPT)

    def test_real_telegram_document_handler_includes_review_policy(self):
        source = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        handler = next(node for node in tree.body
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "handle_document")
        code = ast.unparse(handler)
        self.assertIn("DOCUMENT_REVIEW_PROMPT", code)
        self.assertIn("is_private_owner_update(update)", code)
        self.assertLess(code.index("is_private_owner_update(update)"),
                        code.index("get_file("))
        # The file stays treated as data, never permission to execute commands.
        self.assertIn("Jangan dakwa", code)

    async def test_pdf_api_gets_authenticity_guidance_not_source_rejection(self):
        calls=[]
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"candidates": [
                {"content":{"parts":[{"text":"Dokumen boleh dibaca; kesahihan belum disahkan."}]}}
            ]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            reply = await analyze_pdf(b"%PDF-1.7 fake", "Baca maklumat",
                                      "key", "gemini-3.5-flash-lite", client=client)
        self.assertIn("kesahihan belum disahkan",reply)
        instruction=calls[0]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("jangan menolak",instruction)
        self.assertIn("Kesahihan rasmi",instruction)

    async def test_visual_document_must_check_legibility_and_authenticity(self):
        calls=[]
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices":[
                {"message":{"content":"Dokumen boleh dibaca; belum sah secara bebas."}}
            ]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            text=await analyze_image(b"\x89PNG\r\n\x1a\nfake", "Apa kandungannya?", "key", client=client)
        self.assertIn("belum sah",text)
        instruction=calls[0]["messages"][0]["content"]
        self.assertIn("ringkaskan",instruction)
        self.assertIn("belum disahkan",instruction)


if __name__ == "__main__":
    unittest.main()
