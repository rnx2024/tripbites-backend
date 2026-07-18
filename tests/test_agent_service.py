import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.agent.agent_service import run_agent
from tests.support.redis_fakes import FakeRedis


class AgentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._redis_patcher = patch("app.session.session_store.redis", new=FakeRedis())
        self._redis_patcher.start()
        await asyncio.sleep(0)

    async def asyncTearDown(self) -> None:
        self._redis_patcher.stop()
        await asyncio.sleep(0)

    @staticmethod
    def _brief(
        place: str,
        *,
        weather_text: str = "Clear",
        final: str | None = None,
        risk_level: str = "low",
        travel_advice: list[str] | None = None,
        sources: list[dict[str, str]] | None = None,
        weather_reasons: list[str] | None = None,
        news_reasons: list[str] | None = None,
        news_items: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "place": place,
            "final": final or f"{place} looks generally fine for travel today.",
            "risk_level": risk_level,
            "travel_advice": travel_advice or [],
            "sources": sources or [{"type": "weather"}, {"type": "news"}],
            "weather_summary": {"current": {"weather_text": weather_text}, "day": {}},
            "weather_reasons": weather_reasons or [],
            "news_reasons": news_reasons or [],
            "news_items": news_items or [],
        }

    async def _run_origin_reply_flow(
        self,
        *,
        session_id: str,
        place: str,
        question: str,
        last_exchange: tuple[str | None, str | None],
        pending_context: dict[str, str] | None,
        pending_question: str | None,
        journey_final: str,
        resolve_answer_mode: AsyncMock | None = None,
        recent_turns: list[dict[str, str]] | None = None,
    ):
        journey_payload = {
            "place": place,
            "final": journey_final,
            "risk_level": None,
            "travel_advice": [],
            "sources": [{"type": "weather"}, {"type": "news"}],
        }

        answer_mode_patch = resolve_answer_mode or AsyncMock(return_value="journey_planning")
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=last_exchange)):
            with patch("app.agent.agent_service.get_recent_turns", new=AsyncMock(return_value=recent_turns or [])):
                with patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value=place)):
                    with patch("app.agent.agent_service.get_active_origin", new=AsyncMock(return_value=None)):
                        with patch(
                            "app.agent.agent_service.get_pending_agent_context",
                            new=AsyncMock(return_value=pending_context),
                        ):
                            with patch(
                                "app.agent.agent_service.get_pending_journey_question",
                                new=AsyncMock(return_value=pending_question),
                            ):
                                with patch("app.agent.agent_service._resolve_answer_mode", new=answer_mode_patch):
                                    with patch(
                                        "app.agent.agent_service.set_pending_agent_context",
                                        new=AsyncMock(return_value=None),
                                    ) as clear_context_mock:
                                        with patch(
                                            "app.agent.agent_service.set_pending_journey_question",
                                            new=AsyncMock(return_value=None),
                                        ) as clear_mock:
                                            with patch(
                                                "app.agent.agent_service.mark_tools_called",
                                                new=AsyncMock(return_value=None),
                                            ):
                                                with patch(
                                                    "app.agent.agent_service._answer_journey_question",
                                                    new=AsyncMock(return_value=journey_payload),
                                                ) as journey_mock:
                                                    result = await run_agent(
                                                        session_id=session_id,
                                                        place=place,
                                                        question=question,
                                                    )

        return result, journey_mock, clear_context_mock, clear_mock, answer_mode_patch

    def _assert_journey_call(
        self,
        journey_mock: AsyncMock,
        *,
        place: str,
        question: str,
        origin: str,
        latest_user_message: str,
        pending_question: str | None,
        conversation_history: list[dict[str, str]],
    ) -> None:
        journey_mock.assert_awaited_once_with(
            unittest.mock.ANY,
            place,
            question,
            origin,
            route_or_transport=True,
            latest_user_message=latest_user_message,
            conversation_history=conversation_history,
            pending_question=pending_question,
        )

    async def test_same_destination_followup_uses_general_qa_even_without_recent_turns(self) -> None:
        with patch(
            "app.agent.agent_service.get_last_exchange",
            new=AsyncMock(
                return_value=(
                    "Is it a good idea to travel there this weekend?",
                    "Traveling to Boracay this weekend carries a medium risk level.",
                )
            ),
        ):
            with patch("app.agent.agent_service.get_recent_turns", new=AsyncMock(return_value=[])):
                with patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value="Boracay")):
                    with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                        with patch(
                            "app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)
                        ):
                            with patch(
                                "app.agent.agent_service._resolve_answer_mode",
                                new=AsyncMock(return_value="travel_brief"),
                            ):
                                with patch(
                                    "app.agent.agent_service._answer_general_followup",
                                    new=AsyncMock(
                                        return_value={
                                            "place": "Boracay",
                                            "final": "I don't see anything in the current reporting that spells out a specific route impact.",
                                            "risk_level": None,
                                            "travel_advice": [],
                                            "sources": [{"type": "weather"}, {"type": "news"}],
                                        }
                                    ),
                                ) as general_mock:
                                    with patch("app.agent.agent_service._get_react_app") as react_mock:
                                        with patch(
                                            "app.agent.agent_service.mark_tools_called",
                                            new=AsyncMock(return_value=None),
                                        ):
                                            with patch(
                                                "app.agent.agent_service.set_active_destination",
                                                new=AsyncMock(return_value=None),
                                            ):
                                                result = await run_agent(
                                                    session_id="session-boracay-followup",
                                                    place="Boracay",
                                                    question="What does that mean for travelers?",
                                                )

        self.assertIn("current reporting", result["final"].lower())
        general_mock.assert_awaited_once()
        react_mock.assert_not_called()

    async def test_same_destination_session_followup_uses_general_qa_not_broad_agent(self) -> None:
        with (
            patch(
                "app.agent.agent_service.get_last_exchange",
                new=AsyncMock(
                    return_value=(
                        "Is it a good idea to go there this weekend?",
                        "Traveling to Subic this weekend carries a medium risk level.",
                    )
                ),
            ),
            patch(
                "app.agent.agent_service.get_recent_turns",
                new=AsyncMock(
                    return_value=[
                        {
                            "user": "Is it a good idea to go there this weekend?",
                            "assistant": "Traveling to Subic this weekend carries a medium risk level.",
                        }
                    ]
                ),
            ),
            patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value="Subic")),
        ):
            with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)):
                    with patch(
                        "app.agent.agent_service._resolve_answer_mode",
                        new=AsyncMock(return_value="travel_brief"),
                    ):
                        with patch(
                            "app.agent.agent_service._answer_general_followup",
                            new=AsyncMock(
                                return_value={
                                    "place": "Subic",
                                    "final": "The report doesn't specify which part of Subic was involved.",
                                    "risk_level": None,
                                    "travel_advice": [],
                                    "sources": [{"type": "weather"}, {"type": "news"}],
                                }
                            ),
                        ) as general_mock:
                            with patch("app.agent.agent_service._get_react_app") as react_mock:
                                with patch(
                                    "app.agent.agent_service.mark_tools_called",
                                    new=AsyncMock(return_value=None),
                                ):
                                    with patch(
                                        "app.agent.agent_service.set_active_destination",
                                        new=AsyncMock(return_value=None),
                                    ):
                                        result = await run_agent(
                                            session_id="session-subic-followup",
                                            place="Subic",
                                            question="what part was teh child exploitation case?",
                                        )

        self.assertIn("doesn't specify which part", result["final"].lower())
        general_mock.assert_awaited_once()
        react_mock.assert_not_called()

    async def test_new_destination_does_not_reuse_previous_followup_lock(self) -> None:
        messages = [unittest.mock.Mock()]
        messages[0].content = "Manila looks generally fine for travel."
        app_mock = AsyncMock(return_value={"messages": messages})

        with (
            patch(
                "app.agent.agent_service.get_last_exchange",
                new=AsyncMock(return_value=("What part was the case?", "It wasn't specified.")),
            ),
            patch(
                "app.agent.agent_service.get_recent_turns",
                new=AsyncMock(return_value=[{"user": "What part was the case?", "assistant": "It wasn't specified."}]),
            ),
            patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value="Subic")),
        ):
            with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                with patch("app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)):
                    with patch("app.agent.agent_service.should_include", new=AsyncMock(return_value=(True, True))):
                        with patch(
                            "app.agent.agent_service._resolve_answer_mode",
                            new=AsyncMock(return_value="travel_brief"),
                        ):
                            with patch(
                                "app.agent.agent_service._get_react_app",
                                return_value=unittest.mock.Mock(ainvoke=app_mock),
                            ) as react_mock:
                                with patch(
                                    "app.agent.agent_service.mark_tools_called",
                                    new=AsyncMock(return_value=None),
                                ):
                                    with patch(
                                        "app.agent.agent_service.set_active_destination",
                                        new=AsyncMock(return_value=None),
                                    ):
                                        with patch(
                                            "app.agent.agent_service.set_pending_agent_context",
                                            new=AsyncMock(return_value=None),
                                        ):
                                            with patch(
                                                "app.agent.agent_service.set_pending_journey_question",
                                                new=AsyncMock(return_value=None),
                                            ):
                                                await run_agent(
                                                    session_id="session-new-destination",
                                                    place="Manila",
                                                    question="Is it a good idea to go there this weekend?",
                                                )

        react_mock.assert_called_once()

    async def test_recent_conversation_is_included_in_broad_agent_prompt(self) -> None:
        messages = [unittest.mock.Mock()]
        messages[0].content = "Travel looks fine."
        app_mock = AsyncMock(return_value={"messages": messages})

        with (
            patch(
                "app.agent.agent_service.get_last_exchange",
                new=AsyncMock(return_value=("Should I go there?", "It looks fine.")),
            ),
            patch(
                "app.agent.agent_service.get_recent_turns",
                new=AsyncMock(
                    return_value=[
                        {"user": "Should I go there?", "assistant": "It looks fine."},
                        {"user": "Any disruptions?", "assistant": "No major disruptions reported."},
                    ]
                ),
            ),
            patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)),
            patch("app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)),
            patch("app.agent.agent_service.should_include", new=AsyncMock(return_value=(True, True))),
            patch(
                "app.agent.agent_service._resolve_answer_mode",
                new=AsyncMock(return_value="travel_brief"),
            ),
            patch(
                "app.agent.agent_service._get_react_app",
                return_value=unittest.mock.Mock(ainvoke=app_mock),
            ),
            patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)),
        ):
            await run_agent(
                session_id="session-context",
                place="Cebu",
                question="What are the local events?",
            )

        prompt = app_mock.await_args.args[0]["messages"][0]["content"]
        self.assertIn("Recent conversation context", prompt)
        self.assertIn("Any disruptions?", prompt)
        self.assertIn("No major disruptions reported.", prompt)

    async def test_broad_agent_prompt_does_not_treat_chat_place_mentions_as_destination_switch(self) -> None:
        messages = [unittest.mock.Mock()]
        messages[0].content = "Travel looks fine."
        app_mock = AsyncMock(return_value={"messages": messages})

        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.get_recent_turns", new=AsyncMock(return_value=[])):
                with patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value=None)):
                    with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                        with patch(
                            "app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)
                        ):
                            with patch(
                                "app.agent.agent_service.should_include", new=AsyncMock(return_value=(True, True))
                            ):
                                with patch(
                                    "app.agent.agent_service._resolve_answer_mode",
                                    new=AsyncMock(return_value="travel_brief"),
                                ):
                                    with patch(
                                        "app.agent.agent_service._get_react_app",
                                        return_value=unittest.mock.Mock(ainvoke=app_mock),
                                    ):
                                        with patch(
                                            "app.agent.agent_service.mark_tools_called",
                                            new=AsyncMock(return_value=None),
                                        ):
                                            with patch(
                                                "app.agent.agent_service.set_active_destination",
                                                new=AsyncMock(return_value=None),
                                            ):
                                                await run_agent(
                                                    session_id="session-broad-prompt",
                                                    place="Boracay",
                                                    question="I am from Ilocos Sur and want to know if Boracay is okay this weekend.",
                                                )

        prompt = app_mock.await_args.args[0]["messages"][0]["content"]
        self.assertIn("selected location from the request is the only destination", prompt.lower())
        self.assertNotIn("change the location", prompt.lower())

    async def test_news_followup_hides_brief_metadata(self) -> None:
        initial_items = [
            {
                "title": "PISTON strike affects transport routes",
                "snippet": "The report does not specify whether the strike is in Vigan.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/piston-strike",
            }
        ]
        targeted_items = [
            {
                "title": "PISTON strike update",
                "snippet": "The updated report still does not confirm that the strike is in Vigan.",
                "source": "Local News",
                "date": "2026-03-16T06:00:00+00:00",
                "link": "https://example.com/piston-strike-update",
            }
        ]
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=(initial_items, "")):
                    with patch("app.agent.followup_qa.search_news", return_value=(targeted_items, "")) as search_mock:
                        with patch(
                            "app.agent.followup_qa._plan_followup_action",
                            new=AsyncMock(
                                return_value={
                                    "answered": True,
                                    "answer": "The retrieved reporting does not confirm that the PISTON strike is in Vigan.",
                                    "search_query": "",
                                }
                            ),
                        ):
                            with patch("app.agent.agent_service._get_react_app") as react_mock:
                                result = await run_agent(
                                    session_id="session-1",
                                    place="Vigan",
                                    question="Can you check if the PISTON strike is in Vigan?",
                                )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [{"type": "news"}])
        self.assertIn("confirms the piston strike is in vigan", result["final"].lower())
        self.assertNotIn("the retrieved reporting does not confirm", result["final"].lower())
        search_mock.assert_not_called()
        react_mock.assert_not_called()

    async def test_news_followup_softens_robotic_language(self) -> None:
        initial_items = [
            {
                "title": "Hit-and-run chase reported in La Union",
                "snippet": "Initial reporting mentions an incident in the area.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/hit-and-run",
            }
        ]
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=(initial_items, "")):
                    with patch(
                        "app.agent.followup_qa._plan_followup_action",
                        new=AsyncMock(
                            return_value={
                                "answered": True,
                                "answer": (
                                    "The retrieved reporting does not specify any possible disruptions. "
                                    "For more details, you can check the news article here."
                                ),
                                "search_query": "",
                            }
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-tone-news",
                            place="La Union",
                            question="I plan to go here by Saturday? Any possible disruptions?",
                        )

        self.assertIn("i don't see any confirmed disruptions", result["final"].lower())
        self.assertNotIn("the retrieved reporting does not specify", result["final"].lower())

    async def test_journey_question_without_origin_asks_for_clarification(self) -> None:
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                with patch(
                    "app.agent.agent_service.set_pending_journey_question", new=AsyncMock(return_value=None)
                ) as pending_mock:
                    with patch(
                        "app.agent.agent_service.set_pending_agent_context", new=AsyncMock(return_value=None)
                    ) as context_mock:
                        with patch(
                            "app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)
                        ) as mark_mock:
                            result = await run_agent(
                                session_id="session-2",
                                place="Vigan",
                                question="Should I continue my trip?",
                            )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [])
        self.assertIn("where are you traveling from", result["final"].lower())
        context_mock.assert_awaited_once_with(
            "session-2",
            {
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": "Should I continue my trip?",
                "destination": "Vigan",
            },
        )
        pending_mock.assert_awaited_once_with("session-2", "Should I continue my trip?")
        mark_mock.assert_awaited()

    async def test_journey_origin_reply_resumes_pending_transport_question_without_last_user(self) -> None:
        result, journey_mock, clear_context_mock, clear_mock, _ = await self._run_origin_reply_flow(
            session_id="session-pending-origin",
            place="Davao",
            question="Ilocos Sur",
            last_exchange=(None, "Where are you traveling from?"),
            pending_context={
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": "So what's the best transpo for me? Ferry or plane?",
                "destination": "Davao",
            },
            pending_question="So what's the best transpo for me? Ferry or plane?",
            journey_final="From Ilocos Sur, a flight is likely more practical than a ferry from the data gathered so far.",
        )

        self.assertIn("flight is likely more practical", result["final"].lower())
        clear_context_mock.assert_awaited_once_with("session-pending-origin", None)
        clear_mock.assert_awaited_once_with("session-pending-origin", None)
        self._assert_journey_call(
            journey_mock,
            place="Davao",
            question="So what's the best transpo for me? Ferry or plane?",
            origin="Ilocos Sur",
            latest_user_message="Ilocos Sur",
            conversation_history=[],
            pending_question="So what's the best transpo for me? Ferry or plane?",
        )

    async def test_journey_origin_reply_resumes_previous_transport_question(self) -> None:
        result, journey_mock, clear_context_mock, clear_mock, _ = await self._run_origin_reply_flow(
            session_id="session-journey-origin",
            place="La Union",
            question="Ilocos Sur",
            last_exchange=("What's the best transport to go there?", "Where are you traveling from?"),
            pending_context={
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": "What's the best transport to go there?",
                "destination": "La Union",
            },
            pending_question=None,
            journey_final="Land travel looks practical right now.",
        )

        self.assertEqual(result["final"], "Land travel looks practical right now.")
        clear_context_mock.assert_awaited_once_with("session-journey-origin", None)
        clear_mock.assert_awaited_once_with("session-journey-origin", None)
        self._assert_journey_call(
            journey_mock,
            place="La Union",
            question="What's the best transport to go there?",
            origin="Ilocos Sur",
            latest_user_message="Ilocos Sur",
            conversation_history=[],
            pending_question="What's the best transport to go there?",
        )

    async def test_journey_origin_clarification_sentence_resumes_pending_question(self) -> None:
        result, journey_mock, _, _, _ = await self._run_origin_reply_flow(
            session_id="session-journey-clarify",
            place="Cebu",
            question="I mean I will be coming from Ilocos Sur to Cebu, you asked where I will come from",
            last_exchange=(None, "Where are you traveling from?"),
            pending_context={
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": "Okay, so which is best to travel there, by plane or a ferry?",
                "destination": "Cebu",
            },
            pending_question=None,
            journey_final="From Ilocos Sur, a flight is more practical than a ferry from the data gathered so far.",
        )

        self.assertIn("flight is more practical", result["final"].lower())
        self._assert_journey_call(
            journey_mock,
            place="Cebu",
            question="Okay, so which is best to travel there, by plane or a ferry?",
            origin="Ilocos Sur",
            latest_user_message="I mean I will be coming from Ilocos Sur to Cebu, you asked where I will come from",
            conversation_history=[],
            pending_question="Okay, so which is best to travel there, by plane or a ferry?",
        )

    async def test_origin_only_reply_uses_last_user_transport_question_without_pending_context(self) -> None:
        last_user_question = "So what is the best transpo to get there?"

        async def inspect_mode(*, question, last_reply, recent_turns, pending_agent_context, place):
            await asyncio.sleep(0)
            self.assertEqual(question, last_user_question)
            return "journey_planning"

        result, journey_mock, _, _, _ = await self._run_origin_reply_flow(
            session_id="session-origin-only",
            place="Vigan",
            question="I am from Manila",
            last_exchange=(
                last_user_question,
                "Travel to Vigan is currently rated as LOW risk with clear weather.",
            ),
            pending_context=None,
            pending_question=None,
            journey_final="From Manila, land travel is practical.",
            resolve_answer_mode=inspect_mode,
        )

        self.assertIn("from manila", result["final"].lower())
        self._assert_journey_call(
            journey_mock,
            place="Vigan",
            question=last_user_question,
            origin="Manila",
            latest_user_message="I am from Manila",
            conversation_history=[],
            pending_question=None,
        )

    async def test_journey_transport_fallback_uses_gathered_data_wording(self) -> None:
        brief = self._brief("La Union", weather_text="Broken clouds")
        with (
            patch("app.agent.followup_qa.build_travel_brief", return_value=(brief, "")),
            patch(
                "app.agent.followup_qa._plan_journey_action",
                new=AsyncMock(return_value={"answered": False, "answer": "", "search_query": ""}),
            ),
            patch("app.agent.followup_qa.search_news", return_value=([], "")),
        ):
            with patch("app.agent.followup_qa._run_journey_reasoner", new=AsyncMock(return_value="")):
                result = await run_agent(
                    session_id="session-journey-fallback",
                    place="La Union",
                    question="What's the best transport from Ilocos Sur to get there?",
                )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertIn("data i gathered so far", result["final"].lower())

    async def test_journey_uses_current_evidence_before_search(self) -> None:
        destination_brief = self._brief("Cebu")
        origin_brief = self._brief(
            "Ilocos Sur",
            final="Ilocos Sur looks generally fine for departure today.",
        )
        with (
            patch(
                "app.agent.followup_qa.build_travel_brief", side_effect=[(destination_brief, ""), (origin_brief, "")]
            ),
            patch(
                "app.agent.followup_qa._plan_journey_action",
                new=AsyncMock(
                    return_value={
                        "answered": True,
                        "answer": "From the current conditions, a flight looks more practical than a ferry.",
                        "search_query": "",
                    }
                ),
            ),
            patch("app.agent.followup_qa.search_news", return_value=([], "")) as search_mock,
            patch("app.agent.followup_qa._run_journey_reasoner", new=AsyncMock(return_value="")) as reasoner_mock,
        ):
            result = await run_agent(
                session_id="session-journey-no-search",
                place="Cebu",
                question="Should I take a ferry or plane from Ilocos Sur?",
            )

        self.assertIn("flight looks more practical", result["final"].lower())
        search_mock.assert_not_called()
        reasoner_mock.assert_not_awaited()

    async def test_journey_answer_appends_source_link_when_reasoner_mentions_article(self) -> None:
        brief = self._brief(
            "La Union",
            weather_text="Broken clouds",
            news_items=[
                {
                    "title": "Roadworks expected near San Fernando",
                    "snippet": "Minor roadworks may affect part of the corridor.",
                    "link": "https://example.com/roadworks",
                }
            ],
        )
        with (
            patch("app.agent.followup_qa.build_travel_brief", return_value=(brief, "")),
            patch(
                "app.agent.followup_qa._plan_journey_action",
                new=AsyncMock(return_value={"answered": False, "answer": "", "search_query": ""}),
            ),
            patch("app.agent.followup_qa.search_news", return_value=([], "")),
            patch(
                "app.agent.followup_qa._run_journey_reasoner",
                new=AsyncMock(return_value="You may want to check the article for the latest roadworks details."),
            ),
        ):
            result = await run_agent(
                session_id="session-journey-link",
                place="La Union",
                question="What's the best transport from Ilocos Sur to get there?",
            )

        self.assertIn("source: https://example.com/roadworks", result["final"].lower())

    async def test_journey_transport_search_uses_user_question_terms(self) -> None:
        brief = self._brief("Davao", weather_text="Broken clouds")
        with (
            patch("app.agent.followup_qa.build_travel_brief", return_value=(brief, "")),
            patch(
                "app.agent.followup_qa._plan_journey_action",
                new=AsyncMock(return_value={"answered": False, "answer": "", "search_query": ""}),
            ),
            patch("app.agent.followup_qa.search_news", return_value=([], "")) as search_mock,
        ):
            with patch("app.agent.followup_qa._run_journey_reasoner", new=AsyncMock(return_value="")):
                await run_agent(
                    session_id="session-journey-query",
                    place="Davao",
                    question="Should I take a ferry or plane from Ilocos Sur?",
                )

        search_query = search_mock.call_args.args[0]
        self.assertIn("ferry", search_query.lower())
        self.assertIn("plane", search_query.lower())
        self.assertIn("ilocos", search_query.lower())
        self.assertIn("davao", search_query.lower())

    async def test_news_followup_appends_link_for_relevant_current_item(self) -> None:
        initial_items = [
            {
                "title": "City hall announces weekend market hours",
                "snippet": "Officials shared market hours for the weekend.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/market-hours",
            },
            {
                "title": "Residential fire in Cebu prompts road closure",
                "snippet": "A fire response temporarily affected traffic near the area.",
                "source": "Local News",
                "date": "2026-03-16T06:00:00+00:00",
                "link": "https://example.com/cebu-fire",
            },
        ]
        prior_reply = "Recent local reporting mentions a fire response and some local traffic impact in Cebu."
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, prior_reply))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=(initial_items, "")):
                    with patch(
                        "app.agent.followup_qa._plan_followup_action",
                        new=AsyncMock(
                            return_value={
                                "answered": True,
                                "answer": "You can check the article here for the latest fire-related traffic details.",
                                "search_query": "",
                            }
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-current-link",
                            place="Cebu",
                            question="How does the fire affect travel there?",
                        )

        self.assertIn("source: https://example.com/cebu-fire", result["final"].lower())

    async def test_news_followup_duration_uses_targeted_search_and_direct_answer(self) -> None:
        initial_items = [
            {
                "title": "La Union city launches free vaccination campaign to prevent rabies",
                "snippet": "The initial item announces the campaign but does not include an end date.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/vaccine-campaign",
            }
        ]
        targeted_items = [
            {
                "title": "Vaccination campaign runs through Saturday in San Fernando",
                "snippet": "City officials said the free vaccination drive will continue through Saturday afternoon.",
                "source": "Local News",
                "date": "2026-03-16T06:00:00+00:00",
                "link": "https://example.com/vaccine-campaign-saturday",
            }
        ]
        prior_reply = (
            "San Fernando La Union has a low travel risk level. Recent local reporting highlights "
            "La Union city launches free vaccination campaign to prevent rabies."
        )
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, prior_reply))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=(initial_items, "")):
                    with patch("app.agent.followup_qa.search_news", return_value=(targeted_items, "")) as search_mock:
                        with patch(
                            "app.agent.followup_qa._plan_followup_action",
                            new=AsyncMock(
                                return_value={
                                    "answered": False,
                                    "answer": "",
                                    "search_query": "vaccination saturday san fernando la union",
                                }
                            ),
                        ):
                            with patch(
                                "app.agent.followup_qa._run_followup_reasoner",
                                new=AsyncMock(
                                    return_value="Based on the retrieved update, the vaccination drive is scheduled to continue through Saturday afternoon."
                                ),
                            ):
                                result = await run_agent(
                                    session_id="session-3",
                                    place="San Fernando La Union",
                                    question="I plan to visit on Saturday. Will the vaccination last until Saturday?",
                                )

        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        self.assertEqual(result["sources"], [{"type": "news"}])
        self.assertIn("through saturday afternoon", result["final"].lower())
        search_mock.assert_called_once()
        search_query = search_mock.call_args.args[0]
        self.assertIn("vaccination", search_query.lower())
        self.assertIn("saturday", search_query.lower())

    async def test_news_followup_condenses_generic_recap_when_direct_answer_is_present(self) -> None:
        initial_items = [
            {
                "title": "IRONMAN 70.3 Davao event draws local and foreign athletes",
                "snippet": "The event report does not include an end date in the initial item.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/ironman-davao",
            }
        ]
        prior_reply = (
            "Davao currently presents a low travel risk, with generally favorable conditions for a weekend trip. "
            "However, be aware that the IRONMAN 70.3 Davao event is taking place, which may affect local plans and traffic."
        )
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, prior_reply))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=(initial_items, "")):
                    with patch("app.agent.followup_qa.search_news", return_value=([], "")):
                        with patch(
                            "app.agent.followup_qa._plan_followup_action",
                            new=AsyncMock(
                                return_value={
                                    "answered": False,
                                    "answer": "",
                                    "search_query": "IRONMAN 70.3 Davao weekend schedule",
                                }
                            ),
                        ):
                            with patch(
                                "app.agent.followup_qa._run_followup_reasoner",
                                new=AsyncMock(
                                    return_value=(
                                        "Davao looks generally fine for travel today, with a low risk level. "
                                        "Recent local reporting highlights the IRONMAN 70.3 Davao event. "
                                        "The current reporting does not say whether it will still be running this weekend."
                                    )
                                ),
                            ):
                                result = await run_agent(
                                    session_id="session-condense-news",
                                    place="Davao",
                                    question="Do you know how long IRONMAN will last? Will it be until this weekend?",
                                )

        self.assertIn("does not say whether it will still be running this weekend", result["final"].lower())
        self.assertNotIn("looks generally fine for travel today", result["final"].lower())
        self.assertNotIn("low risk level", result["final"].lower())
        self.assertEqual(result["final"].count("."), 1)

    async def test_news_followup_appends_source_link_when_answer_references_article(self) -> None:
        initial_items = [
            {
                "title": "Hit-and-run chase reported in La Union",
                "snippet": "Initial reporting mentions an incident in the area.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/hit-and-run",
            }
        ]
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=(initial_items, "")):
                    with patch(
                        "app.agent.followup_qa._plan_followup_action",
                        new=AsyncMock(
                            return_value={
                                "answered": True,
                                "answer": (
                                    "The retrieved reporting does not specify any possible disruptions. "
                                    "For more details, you can check the news article here."
                                ),
                                "search_query": "",
                            }
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-4",
                            place="La Union",
                            question="I plan to go here by Saturday? Any possible disruptions?",
                        )

        self.assertIn("source: https://example.com/hit-and-run", result["final"].lower())

    async def test_weather_followup_softens_robotic_language(self) -> None:
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, None))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch(
                    "app.agent.followup_qa.get_weather_summary",
                    return_value=("Broken clouds with a temperature of 25.78C.", ""),
                ):
                    with patch(
                        "app.agent.followup_qa._run_followup_reasoner",
                        new=AsyncMock(
                            return_value=(
                                "The retrieved weather data for La Union shows broken clouds with a temperature of 25.78C, "
                                "but it does not specify any possible risks or weather disturbances."
                            )
                        ),
                    ):
                        result = await run_agent(
                            session_id="session-tone-weather",
                            place="La Union",
                            question="So no possible risks? or weather disturbances?",
                        )

        self.assertIn("the current weather for la union", result["final"].lower())
        self.assertIn("doesn't point to any specific weather disruptions right now", result["final"].lower())
        self.assertNotIn("the retrieved weather data", result["final"].lower())

    async def test_news_followup_no_answer_found_fallback_is_direct(self) -> None:
        prior_reply = "Recent local reporting mentions a weekend event in Cebu."
        with patch("app.agent.agent_service.get_last_exchange", new=AsyncMock(return_value=(None, prior_reply))):
            with patch("app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)):
                with patch("app.agent.followup_qa.get_news_items", return_value=([], "")):
                    with patch("app.agent.followup_qa.search_news", return_value=([], "")):
                        with patch(
                            "app.agent.followup_qa._plan_followup_action",
                            new=AsyncMock(
                                return_value={
                                    "answered": False,
                                    "answer": "",
                                    "search_query": "cebu weekend event running",
                                }
                            ),
                        ):
                            with patch("app.agent.followup_qa._run_followup_reasoner", new=AsyncMock(return_value="")):
                                result = await run_agent(
                                    session_id="session-no-answer-news",
                                    place="Cebu",
                                    question="Will that event still be running this weekend?",
                                )

        self.assertEqual(result["final"], "I couldn't find a confirmed answer in the current news for Cebu.")

    async def test_general_followup_does_not_pass_concern_summary_into_qa_evidence(self) -> None:
        brief = self._brief(
            "Batanes",
            weather_text="Overcast",
            final="Batanes looks fine for travel this weekend.",
            sources=[{"type": "weather"}],
            travel_advice=["Stay hydrated."],
            weather_reasons=["Warm temperatures."],
        )

        async def inspect_plan(_llm, *, place, question, evidence):
            await asyncio.sleep(0)
            self.assertEqual(place, "Batanes")
            self.assertEqual(question, "Do you have a list of resorts with aircon?")
            place_evidence = evidence["place_evidence"]
            self.assertNotIn("risk_level", place_evidence)
            self.assertNotIn("final", place_evidence)
            self.assertNotIn("travel_advice", place_evidence)
            return {"answered": False, "answer": "", "search_query": "Batanes resorts with aircon"}

        targeted_items = [
            {
                "title": "Batanes resort with air-conditioned rooms",
                "snippet": "A local resort listing mentions air-conditioned rooms in Basco.",
                "link": "https://example.com/batanes-aircon",
            }
        ]

        with patch(
            "app.agent.agent_service.get_last_exchange",
            new=AsyncMock(return_value=(None, "Traveling to Batanes this weekend is considered a good idea.")),
        ):
            with patch(
                "app.agent.agent_service.get_recent_turns",
                new=AsyncMock(
                    return_value=[
                        {
                            "user": "Is going there this weekend a good idea?",
                            "assistant": "Traveling to Batanes this weekend is considered a good idea.",
                        }
                    ]
                ),
            ):
                with patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value="Batanes")):
                    with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                        with patch(
                            "app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)
                        ):
                            with patch(
                                "app.agent.agent_service._resolve_answer_mode",
                                new=AsyncMock(return_value="travel_brief"),
                            ):
                                with patch(
                                    "app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)
                                ):
                                    with patch(
                                        "app.agent.agent_service.set_active_destination",
                                        new=AsyncMock(return_value=None),
                                    ):
                                        with patch(
                                            "app.agent.followup_qa.build_travel_brief", return_value=(brief, "")
                                        ):
                                            with patch("app.agent.followup_qa._plan_followup_action", new=inspect_plan):
                                                with patch(
                                                    "app.agent.followup_qa.search_news",
                                                    return_value=(targeted_items, ""),
                                                ) as search_mock:
                                                    with patch(
                                                        "app.agent.followup_qa._run_followup_reasoner",
                                                        new=AsyncMock(
                                                            return_value="Yes. I found at least one resort listing in Batanes that mentions air-conditioned rooms. Source: https://example.com/batanes-aircon"
                                                        ),
                                                    ):
                                                        result = await run_agent(
                                                            session_id="session-batanes-aircon",
                                                            place="Batanes",
                                                            question="Do you have a list of resorts with aircon?",
                                                        )

        self.assertIn("yes.", result["final"].lower())
        self.assertIn("source: https://example.com/batanes-aircon", result["final"].lower())
        self.assertIsNone(result["risk_level"])
        self.assertEqual(result["travel_advice"], [])
        search_mock.assert_called_once()

    async def test_general_followup_resort_question_does_not_fall_back_to_travel_recap(self) -> None:
        brief = self._brief(
            "Batanes",
            weather_text="Overcast",
            final="Batanes looks fine for travel this weekend.",
            sources=[{"type": "weather"}],
            travel_advice=["Stay hydrated."],
            weather_reasons=["Warm temperatures."],
        )
        with patch(
            "app.agent.agent_service.get_last_exchange",
            new=AsyncMock(return_value=(None, "Traveling to Batanes this weekend is considered a good idea.")),
        ):
            with patch(
                "app.agent.agent_service.get_recent_turns",
                new=AsyncMock(
                    return_value=[
                        {
                            "user": "Is going there this weekend a good idea?",
                            "assistant": "Traveling to Batanes this weekend is considered a good idea.",
                        }
                    ]
                ),
            ):
                with patch("app.agent.agent_service.get_active_destination", new=AsyncMock(return_value="Batanes")):
                    with patch("app.agent.agent_service.get_pending_agent_context", new=AsyncMock(return_value=None)):
                        with patch(
                            "app.agent.agent_service.get_pending_journey_question", new=AsyncMock(return_value=None)
                        ):
                            with patch(
                                "app.agent.agent_service._resolve_answer_mode",
                                new=AsyncMock(return_value="travel_brief"),
                            ):
                                with patch(
                                    "app.agent.agent_service.mark_tools_called", new=AsyncMock(return_value=None)
                                ):
                                    with patch(
                                        "app.agent.agent_service.set_active_destination",
                                        new=AsyncMock(return_value=None),
                                    ):
                                        with patch(
                                            "app.agent.followup_qa.build_travel_brief", return_value=(brief, "")
                                        ):
                                            with patch(
                                                "app.agent.followup_qa._plan_followup_action",
                                                new=AsyncMock(
                                                    return_value={
                                                        "answered": False,
                                                        "answer": "",
                                                        "search_query": "Batanes resorts aircon",
                                                    }
                                                ),
                                            ):
                                                with patch("app.agent.followup_qa.search_news", return_value=([], "")):
                                                    with patch(
                                                        "app.agent.followup_qa._run_followup_reasoner",
                                                        new=AsyncMock(
                                                            return_value="I couldn't find a confirmed list of Batanes resorts with air conditioning from the data I gathered so far."
                                                        ),
                                                    ):
                                                        result = await run_agent(
                                                            session_id="session-batanes-resorts",
                                                            place="Batanes",
                                                            question="Do you have a list of resorts?",
                                                        )

        self.assertIn("couldn't find a confirmed list", result["final"].lower())
        self.assertNotIn("risk level", result["final"].lower())
        self.assertNotIn("stay hydrated", result["final"].lower())


if __name__ == "__main__":
    unittest.main()
