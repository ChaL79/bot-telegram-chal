import ast
import io
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from media_tools import (
    MAX_IMAGE_BYTES, MAX_DOCUMENT_BYTES, analyze_image, analyze_pdf,
    detect_image_type, extract_document_text
)


def zip_document(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buf.getvalue()


class MediaTests(unittest.IsolatedAsyncioTestCase):
    def test_image_magic_not_user_filename(self):
        self.assertEqual(detect_image_type(b"\x89PNG\r\n\x1a\nDATA"), "image/png")
        self.assertEqual(detect_image_type(b"\xff\xd8\xffDATA"), "image/jpeg")
        self.assertIsNone(detect_image_type(b"<script>"))
        self.assertIsNone(detect_image_type(b"%PDF-1.7"))

    def test_local_text_and_docx(self):
        self.assertIn("Hello", extract_document_text(b"Hello there", ".txt"))
        docx = zip_document({
            "word/document.xml": (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:body><w:p><w:r>'
                '<w:t>Hello Word</w:t></w:r></w:p></w:body></w:document>'
            )
        })
        self.assertIn("Hello Word", extract_document_text(docx, ".docx"))
        with self.assertRaises(ValueError):
            extract_document_text(b"Z" * (MAX_DOCUMENT_BYTES + 1), ".txt")

    def test_xlsx_and_zip_bomb_guard(self):
        worksheet = (
            '<worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheetData><row r="1">'
            '<c r="A1" t="inlineStr"><is><t>Sales</t></is></c>'
            '<c r="B1"><v>42</v></c></row></sheetData></worksheet>'
        )
        xlsx = zip_document({"xl/worksheets/sheet1.xml": worksheet})
        result = extract_document_text(xlsx, ".xlsx")
        self.assertIn("Sales", result)
        self.assertIn("42", result)
        bomb = zip_document({"word/document.xml": "x" * 3_100_000})
        with self.assertRaises(ValueError):
            extract_document_text(bomb, ".docx")

    async def test_image_request_existing_multimodal_model(self):
        captured = []
        def handler(req):
            captured.append(req)
            return httpx.Response(
                200,
                json={"choices":[{"message":{"content":"Ada satu lampu."}}]})
        img = b"\x89PNG\r\n\x1a\nDATA"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
            answer = await analyze_image(img, "Apa ini?", "test-key", client=cli)
        self.assertIn("lampu", answer)
        payload = __import__("json").loads(captured[0].content)
        self.assertEqual(payload["model"], "qwen/qwen3.8-27b")
        self.assertTrue(payload["messages"][-1]["content"][-1]
                        ["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(captured[0].headers.get("authorization"), "Bearer test-key")

    async def test_pdf_inline_uses_existing_gemini_without_file_store(self):
        captured = []
        def handler(req):
            captured.append(req)
            return httpx.Response(
                200,
                json={"candidates":[{"content":{"parts":[{"text":"Tiga fakta."}]}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as cli:
            answer = await analyze_pdf(b"%PDF-1.7 fake", "Ringkaskan",
                                        "gem-key", "gemini-3.5-flash-lite", client=cli)
        self.assertIn("Tiga fakta", answer)
        payload = __import__("json").loads(captured[0].content)
        self.assertEqual(payload["contents"][0]["parts"][1]["inlineData"]
                         ["mimeType"], "application/pdf")
        self.assertIn("gemini-3.5-flash-lite:generateContent", str(captured[0].url))
        self.assertEqual(captured[0].headers.get("x-goog-api-key"), "gem-key")

    async def test_rejects_bad_or_oversize_media_before_any_network(self):
        self.assertIn("tidak disokong",
                      await analyze_image(b"fake", "?", "key"))
        self.assertIn("4 MB",
                      await analyze_image(b"\x89PNG\r\n\x1a\n" +
                                          b"x" * MAX_IMAGE_BYTES, "?", "key"))
        self.assertIn("tidak sah",
                      await analyze_pdf(b"wrong", "?", "key", "gemini-3.5-flash-lite"))

    def test_production_media_entrypoints_owner_guard_first(self):
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
        for name in ("handle_photo", "handle_document"):
            fn = next(x for x in tree.body if isinstance(x, ast.AsyncFunctionDef)
                      and x.name == name)
            source = ast.unparse(fn)
            self.assertIn("is_private_owner_update(update)", source)
            self.assertLess(source.index("is_private_owner_update(update)"),
                            source.index("get_file("))


if __name__ == "__main__":
    unittest.main()
