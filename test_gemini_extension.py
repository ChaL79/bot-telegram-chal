import unittest
import httpx
from gemini_extension import extract_gemini_text, get_gemini_reply, _safe_model, is_authorized_owner


class GeminiExtensionTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_sends_header_and_extracts_text(self):
        seen = []
        def mock(request):
            seen.append(request)
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Salam!"}]}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            result = await get_gemini_reply("Hi", "TEST_KEY", client=client)
        self.assertEqual(result, "Salam!")
        self.assertEqual(seen[0].headers["x-goog-api-key"], "TEST_KEY")
        self.assertIn("gemini-3.5-flash-lite:generateContent", str(seen[0].url))
        self.assertIn("Hi", seen[0].content.decode())

    async def test_rate_limit_no_retry(self):
        count = 0
        def mock(request):
            nonlocal count
            count += 1
            return httpx.Response(429, json={"error": {"message": "quota"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            result = await get_gemini_reply("Hi", "TEST_KEY", client=client)
        self.assertEqual(count, 1)
        self.assertIn("Had penggunaan", result)

    async def test_missing_key_makes_no_request(self):
        def mock(request):
            raise AssertionError("Must not call API without key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            result = await get_gemini_reply("Hi", "", client=client)
        self.assertIn("belum dikonfigurasikan", result)

    def test_owner_authorization_fail_closed(self):
        self.assertTrue(is_authorized_owner(1234, "1234", "private"))
        self.assertFalse(is_authorized_owner(1234, "", "private"))
        self.assertFalse(is_authorized_owner(1234, "1234", "group"))
        self.assertFalse(is_authorized_owner(999, "1234", "private"))
        self.assertFalse(is_authorized_owner(1234, "abc", "private"))

    async def test_blank_prompt_never_calls_api(self):
        def mock(request):
            raise AssertionError("API should not be called")
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            answer = await get_gemini_reply("  ", "TEST_KEY", client=client)
        self.assertIn("Taip /gemini", answer)

    def test_validate_model_and_response(self):
        self.assertRaises(ValueError, _safe_model, "../../../test")
        self.assertIsNone(extract_gemini_text({"candidates": []}))
        self.assertEqual(
            extract_gemini_text({"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}),
            "A\nB",
        )


if __name__ == "__main__":
    unittest.main()
