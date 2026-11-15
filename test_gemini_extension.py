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

    async def test_history_contents_are_forwarded_in_order(self):
        import json
        from gemini_extension import get_gemini_reply_result
        seen = []
        def mock(request):
            seen.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "MELUR"}]}}]})
        history = [
            {"role": "user", "parts": [{"text": "Ingat MELUR"}]},
            {"role": "model", "parts": [{"text": "Baik, MELUR"}]},
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            answer, success = await get_gemini_reply_result(
                "Apa perkataan tadi?", "TEST_KEY", history=history, client=client,
            )
        self.assertTrue(success)
        self.assertEqual(answer, "MELUR")
        self.assertEqual([x["role"] for x in seen[0]["contents"]], ["user", "model", "user"])
        self.assertEqual(seen[0]["contents"][0]["parts"][0]["text"], "Ingat MELUR")
        self.assertEqual(seen[0]["contents"][-1]["parts"][0]["text"], "Apa perkataan tadi?")

    async def test_api_failure_result_not_successful(self):
        from gemini_extension import get_gemini_reply_result
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(429)
        )) as client:
            answer, success = await get_gemini_reply_result("Hi", "TEST_KEY", client=client)
        self.assertFalse(success)
        self.assertIn("Had penggunaan", answer)

    async def test_invalid_history_blocked_before_api_call(self):
        from gemini_extension import get_gemini_reply_result
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: self.fail("Invalid history must never reach Gemini")
        )) as client:
            with self.assertRaises(ValueError):
                await get_gemini_reply_result(
                    "Hi", "TEST_KEY", history=[{"role": "model", "parts": [{"text": "bad"}]}], client=client,
                )

    def test_validate_model_and_response(self):
        self.assertRaises(ValueError, _safe_model, "../../../test")
        self.assertIsNone(extract_gemini_text({"candidates": []}))
        self.assertEqual(
            extract_gemini_text({"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}),
            "A\nB",
        )


if __name__ == "__main__":
    unittest.main()
