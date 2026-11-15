"""Memory tests: TTL, bounds, isolation, reset, and read-only copies."""
import unittest

from gemini_memory import (
    HISTORY_KEY, IDLE_TTL_SECONDS, MAX_HISTORY_CHARS, MAX_MESSAGES,
    LAST_ACTIVE_KEY, clear_history, read_history, record_exchange,
)


class GeminiMemoryTests(unittest.TestCase):
    def test_melur_followup_history(self):
        state = {}
        record_exchange(state, "Ingat perkataan ujian saya: MELUR.", "Baik, MELUR.", now=100)
        history = read_history(state, now=101)
        self.assertEqual([x["role"] for x in history], ["user", "model"])
        self.assertIn("MELUR", history[0]["parts"][0]["text"])

    def test_expire_at_one_hour_including_cleanup(self):
        state = {}
        record_exchange(state, "Simpan A", "Baik", now=100)
        self.assertEqual(len(read_history(state, now=100 + IDLE_TTL_SECONDS - 1)), 2)
        self.assertEqual(read_history(state, now=100 + IDLE_TTL_SECONDS), [])
        self.assertNotIn(HISTORY_KEY, state)
        self.assertNotIn(LAST_ACTIVE_KEY, state)

    def test_bounded_full_exchanges(self):
        state = {}
        for i in range(20):
            record_exchange(state, f"soalan-{i}", f"jawapan-{i}", now=100+i)
        history = read_history(state, now=120)
        self.assertEqual(len(history), MAX_MESSAGES)
        self.assertEqual(history[0]["parts"][0]["text"], "soalan-15")
        self.assertEqual(history[-1]["parts"][0]["text"], "jawapan-19")
        self.assertEqual([x["role"] for x in history], ["user", "model"] * 5)

    def test_total_size_and_pair_preservation(self):
        state = {}
        for i in range(20):
            record_exchange(state, f"{i}"+("a" * 2000), f"{i}"+("b" * 2000), now=100+i)
        history = read_history(state, now=120)
        self.assertLessEqual(sum(len(x["parts"][0]["text"]) for x in history), MAX_HISTORY_CHARS)
        self.assertEqual(len(history) % 2, 0)
        self.assertEqual(history[-1]["role"], "model")

    def test_read_is_a_copy_and_sessions_isolated(self):
        first, second = {}, {}
        record_exchange(first, "MELUR", "OK", now=0)
        copy = read_history(first, now=1)
        copy[0]["parts"][0]["text"] = "CHANGED"
        self.assertEqual(read_history(first, now=1)[0]["parts"][0]["text"], "MELUR")
        self.assertEqual(read_history(second, now=1), [])

    def test_reset_memory(self):
        state = {}
        record_exchange(state, "Rahsia sementara", "OK", now=10)
        self.assertTrue(clear_history(state))
        self.assertEqual(read_history(state, now=11), [])
        self.assertFalse(clear_history(state))

    def test_malformed_history_is_rejected(self):
        state = {HISTORY_KEY: [{"role": "model", "parts": [{"text": "bad"}]}], LAST_ACTIVE_KEY: 2}
        self.assertEqual(read_history(state, now=3), [])
        self.assertNotIn(HISTORY_KEY, state)


if __name__ == "__main__":
    unittest.main()
