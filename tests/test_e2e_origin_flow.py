import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.agent.agent_policy import classify_answer_mode
from app.main import app
from app.session.session_keys import sess_key
from app.settings import settings
from tests.support.redis_fakes import FakeRedis


class OriginFlowE2ETests(unittest.TestCase):
    def test_origin_only_reply_resumes_transport_question_when_pending_context_is_missing(self) -> None:
        fake_redis = FakeRedis()

        async def resolve_mode_stub(*, question, last_reply, recent_turns, pending_agent_context, place):
            await asyncio.sleep(0)
            _ = (recent_turns, pending_agent_context, place)
            return classify_answer_mode(question, last_reply)

        journey_result = {
            "place": "Vigan",
            "final": "From Manila, land travel is practical.",
            "risk_level": None,
            "travel_advice": [],
            "sources": [{"type": "weather"}, {"type": "news"}],
        }

        with (
            patch("app.redis_client.redis", fake_redis),
            patch("app.session.session_store.redis", fake_redis),
            patch("app.redis_client.init_redis", new=AsyncMock(return_value=None)),
            patch("app.redis_client.close_redis", new=AsyncMock(return_value=None)),
            patch("app.agent.agent_service._resolve_answer_mode", new=resolve_mode_stub),
            patch(
                "app.agent.agent_service._answer_journey_question",
                new=AsyncMock(return_value=journey_result),
            ) as journey_mock,
        ):
            client = TestClient(app)
            api_headers = {"x-api-key": settings.api_key}
            session_resp = client.post("/session", headers=api_headers)
            session_payload = session_resp.json()
            session_id = session_payload["session_id"]
            session_token = session_payload["session_token"]
            chat_headers = {
                "x-api-key": settings.api_key,
                "x-session-id": session_id,
                "x-session-token": session_token,
            }

            first = client.post(
                "/chat",
                headers=chat_headers,
                json={"place": "Vigan", "question": "So what is the best transpo to get there?"},
            )
            self.assertIn("where are you traveling from", first.json()["final"].lower())

            session_key = sess_key(session_id)
            fake_redis.hashes.get(session_key, {}).pop("pending_agent_context", None)
            fake_redis.hashes.get(session_key, {}).pop("pending_journey_question", None)

            second = client.post(
                "/chat",
                headers=chat_headers,
                json={"place": "Vigan", "question": "I am from Manila"},
            )

        self.assertEqual(second.json()["final"], "From Manila, land travel is practical.")
        journey_mock.assert_awaited()
        args, kwargs = journey_mock.await_args
        self.assertEqual(args[1], "Vigan")
        self.assertEqual(args[2], "So what is the best transpo to get there?")
        self.assertEqual(args[3], "Manila")
        self.assertTrue(kwargs.get("route_or_transport"))
        self.assertEqual(kwargs.get("latest_user_message"), "I am from Manila")
        self.assertIsNone(kwargs.get("pending_question"))
        conversation_history = kwargs.get("conversation_history") or []
        self.assertTrue(conversation_history)
        self.assertIn("best transpo", conversation_history[0].get("user", "").lower())


if __name__ == "__main__":
    unittest.main()
