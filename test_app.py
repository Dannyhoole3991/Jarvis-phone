"""
The phone server: Chat tab (quick brain), PC tab (jobs), the dashboard copy, and no double speech.
OpenAI, ElevenLabs and the smart plug are faked; nothing leaves this machine.

    py -3 -m unittest test_app -v
"""

import os
import tempfile
import time
import unittest
from unittest import mock

import app as A

AUTH = {"Authorization": "Bearer test-secret"}


class PhoneServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(A, "STATE_PATH", os.path.join(self.tmp.name, "state.json")),
            mock.patch.object(A, "SHARED_SECRET", "test-secret"),
            mock.patch.object(A, "_synthesize_speech", return_value=b"mp3"),
        ]
        for p in self.patches:
            p.start()
        self.client = A.app.test_client()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def pc_online(self):
        self.client.post("/api/pc_poll", headers=AUTH)

    def events(self):
        return [e["text"] for e in self.client.get("/api/pc_events", headers=AUTH).get_json()["events"]]

    def test_chat_tab_never_waits_on_the_pc(self):
        self.pc_online()
        with mock.patch.object(A, "_ask_chat_brain", return_value="Very well, sir.") as brain:
            r = self.client.post("/api/message", json={"text": "how are you"}, headers=AUTH).get_json()
        self.assertEqual((r["reply"], r["source"]), ("Very well, sir.", "cloud"))
        brain.assert_called_once()
        self.assertEqual(self.client.post("/api/pc_poll", headers=AUTH).get_json()["pending_commands"], [],
                         "nothing was sent to the PC")

    def test_chat_brain_gets_the_dashboard(self):
        self.client.post("/api/dashboard_push", headers=AUTH, json={"store": {"orders": {"ok": True, "orders": [
            {"number": "#1001", "date": "Yesterday", "customer": "Joanne Hoole", "total": "£18.98", "payment": "Paid",
             "fulfilment": "In progress", "items": "2 items"}], "payouts": {"pending": "£18.35", "transactions": [
            {"order": "#1001", "net": "£18.35", "payout_date": "Oct 23, 2026"}]}}, "products": []}})
        with mock.patch.object(A, "_ask_chat_brain", return_value="One order, sir.") as brain:
            self.client.post("/api/message", json={"text": "any orders?"}, headers=AUTH)
        dashboard_text = brain.call_args[0][3]
        self.assertIn("#1001", dashboard_text)
        self.assertIn("£18.35 on Oct 23, 2026", dashboard_text)
        self.assertIn("PC is off", dashboard_text)

    def test_pc_tab_queues_and_returns_at_once(self):
        self.pc_online()
        r = self.client.post("/api/message", json={"text": "open steam", "mode": "pc"}, headers=AUTH).get_json()
        self.assertTrue(r["queued"])
        handed = self.client.post("/api/pc_poll", headers=AUTH).get_json()["pending_commands"]
        self.assertEqual(handed, [{"id": r["queued"], "command": "open steam"}])
        self.assertFalse(self.client.get(f"/api/pc_job/{r['queued']}", headers=AUTH).get_json()["done"])
        self.client.post("/api/pc_said", headers=AUTH, json={"text": "Opening Steam, sir."})
        self.client.post("/api/pc_command_result", headers=AUTH, json={"id": r["queued"], "reply": "Opening Steam, sir."})
        self.assertTrue(self.client.get(f"/api/pc_job/{r['queued']}", headers=AUTH).get_json()["done"])
        self.assertEqual(self.events(), ["Opening Steam, sir."], "said once, not twice")

    def test_answer_is_never_doubled_whichever_arrives_first(self):
        self.pc_online()
        job = self.client.post("/api/message", json={"text": "check tiktok", "mode": "pc"}, headers=AUTH).get_json()["queued"]
        self.client.post("/api/pc_command_result", headers=AUTH, json={"id": job, "reply": "17 views this week, sir."})
        self.client.post("/api/pc_said", headers=AUTH, json={"text": "17 views this week, sir."})
        self.assertEqual(self.events(), ["17 views this week, sir."])

    def test_an_unspoken_answer_still_reaches_the_pc_tab(self):
        self.pc_online()
        job = self.client.post("/api/message", json={"text": "do a thing", "mode": "pc"}, headers=AUTH).get_json()["queued"]
        self.client.post("/api/pc_command_result", headers=AUTH, json={"id": job, "reply": "Jarvis took too long to reply."})
        self.assertEqual(self.events(), ["Jarvis took too long to reply."])

    def test_pc_tab_when_the_pc_is_off(self):
        r = self.client.post("/api/message", json={"text": "open steam", "mode": "pc"}, headers=AUTH).get_json()
        self.assertEqual(r["source"], "pc_offline")
        self.assertIn("offline", r["reply"])

    def test_dashboard_needs_the_key(self):
        self.assertEqual(self.client.post("/api/dashboard_push", json={"x": 1}).status_code, 401)
        self.assertEqual(self.client.get("/api/dashboard").status_code, 401)
        self.client.post("/api/dashboard_push", headers=AUTH, json={"needs_you": {"text": "Pay CJ"}})
        d = self.client.get("/api/dashboard", headers=AUTH).get_json()
        self.assertEqual(d["dashboard"]["needs_you"]["text"], "Pay CJ")
        self.assertGreater(d["at"], time.time() - 5)

    def test_a_slow_chat_reply_does_not_wipe_what_the_pc_said_meanwhile(self):
        def slow_brain(*args):
            self.client.post("/api/pc_said", headers=AUTH, json={"text": "Your video is live, sir."})
            return "Fine, sir."
        with mock.patch.object(A, "_ask_chat_brain", side_effect=slow_brain):
            self.client.post("/api/message", json={"text": "hello"}, headers=AUTH)
        self.assertIn("Your video is live, sir.", self.events())

    def test_compressed_dashboard_is_unpacked(self):
        import base64, gzip, json
        body = {"gz": base64.b64encode(gzip.compress(json.dumps({"needs_you": {"text": "See https://example.com <b>"}}).encode())).decode()}
        self.assertEqual(self.client.post("/api/dashboard_push", headers=AUTH, json=body).status_code, 200)
        d = self.client.get("/api/dashboard", headers=AUTH).get_json()
        self.assertEqual(d["dashboard"]["needs_you"]["text"], "See https://example.com <b>")
        self.assertEqual(self.client.post("/api/dashboard_push", headers=AUTH, json={"gz": "not base64!"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
