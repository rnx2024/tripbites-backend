# app/agent_policy.py
from __future__ import annotations

import re
from typing import Literal

AnswerMode = Literal["travel_brief", "news_followup", "weather_followup", "journey_planning"]


_WEATHER_TERMS = (
    "weather",
    "forecast",
    "temperature",
    "rain",
    "storm",
    "wind",
    "uv",
    "heat",
    "snow",
    "fog",
    "typhoon",
    "hurricane",
    "humid",
    "humidity",
    "clear",
    "cloud",
)

_NEWS_TERMS = (
    "news",
    "headline",
    "headlines",
    "update",
    "updates",
    "disruption",
    "disruptions",
    "incident",
    "incidents",
    "protest",
    "strike",
    "closure",
    "closures",
    "closed",
    "outage",
    "traffic",
    "where",
    "location",
    "locations",
    "area",
    "areas",
    "reported",
    "reporting",
    "article",
)

_TRAVEL_BRIEF_TERMS = (
    "travel brief",
    "travel advice",
    "practical advice",
    "should i go",
    "should we go",
    "fine for travel",
    "safe",
    "risk",
    "go/no-go",
    "worth it",
)

_JOURNEY_TERMS = (
    "continue my trip",
    "continue the trip",
    "continue travelling",
    "continue traveling",
    "on the way",
    "along the way",
    "getting there",
    "get there",
    "travel to ",
    "trip to ",
    "best route",
    "best way",
    "best transpo",
    "best transport",
    "how should i get",
    "how do i get",
    "which route",
    "which transport",
)

_ROUTE_TRANSPORT_TERMS = (
    "best route",
    "best way",
    "best transpo",
    "best transport",
    "which route",
    "which transport",
    "how should i get",
    "how do i get",
    "should i take",
    "go by",
)

_TRANSPORT_MODE_TERMS = (
    "plane",
    "flight",
    "ferry",
    "boat",
    "ship",
    "bus",
    "car",
    "drive",
    "train",
)

_ORIGIN_QUESTION_TERMS = (
    "where are you traveling from",
    "where are you coming from",
    "what is your departure location",
)

_FOLLOWUP_CONTINUATION_TERMS = (
    "last",
    "still",
    "continue",
    "continuing",
    "remain",
    "until",
    "ongoing",
    "on saturday",
    "on sunday",
    "on monday",
    "on tuesday",
    "on wednesday",
    "on thursday",
    "on friday",
    "next week",
    "this weekend",
)

_CONTEXT_REFERENCE_TERMS = (
    "that",
    "this",
    "it",
    "there",
)

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "before",
    "between",
    "campaign",
    "city",
    "could",
    "does",
    "fine",
    "from",
    "generally",
    "going",
    "have",
    "here",
    "into",
    "just",
    "know",
    "last",
    "local",
    "look",
    "looks",
    "make",
    "more",
    "most",
    "need",
    "news",
    "reported",
    "reporting",
    "risk",
    "saturday",
    "should",
    "some",
    "that",
    "there",
    "they",
    "this",
    "those",
    "through",
    "time",
    "today",
    "travel",
    "trip",
    "until",
    "very",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
}


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def is_trip_planning_question(q: str | None) -> bool:
    """
    Heuristic: identify trip/planning go/no-go questions that should trigger both
    weather + news signals even without explicit 'weather'/'news' keywords.
    """
    if not q:
        return False

    s = q.lower()

    trip_terms = (
        "trip",
        "travel",
        "go on",
        "should i go",
        "should we go",
        "visit",
        "outing",
        "hike",
        "beach",
        "roadtrip",
        "commute",
        "drive",
        "fly",
        "ferry",
    )

    date_terms = (
        "today",
        "tomorrow",
        "tonight",
        "weekend",
        "next week",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "next ",
        "this ",
    )

    intent_terms = ("safe", "risk", "cancel", "postpone", "worth it")

    return any(t in s for t in trip_terms) and (any(t in s for t in date_terms) or any(t in s for t in intent_terms))


def decide_tool_includes(question: str | None) -> tuple[bool, bool]:
    """
    Decide whether the agent should be allowed to call weather_tool/news_tool.

    Returns: (include_weather, include_news)
    """
    if not question:
        return True, True

    if is_trip_planning_question(question):
        return True, True

    q = question.lower()

    inc_w = _has_any_term(q, _WEATHER_TERMS)
    inc_n = _has_any_term(q, _NEWS_TERMS)

    return (inc_w, inc_n) if (inc_w or inc_n) else (False, False)


def detect_force_signals(question: str) -> tuple[bool, bool]:
    """
    Decide whether the request should bypass session suppression (force include).
    Trip-planning questions force both.
    """
    q = question or ""
    q_lc = q.lower()

    if is_trip_planning_question(q) or is_journey_planning_question(q):
        return True, True

    return ("weather" in q_lc, "news" in q_lc)


def classify_answer_mode(question: str | None, last_reply: str | None = None) -> AnswerMode:
    if not question:
        return "travel_brief"

    q = question.lower()
    last = (last_reply or "").lower()

    if _should_force_journey(question, q, last, last_reply):
        return "journey_planning"

    has_weather, has_news = _resolve_followup_signals(q, last)

    if has_news and not has_weather:
        return "news_followup"
    if has_weather and not has_news:
        return "weather_followup"
    if is_trip_planning_question(question) or _has_any_term(q, _TRAVEL_BRIEF_TERMS):
        return "travel_brief"
    return "travel_brief"


def _infer_followup_topic(question: str, last_reply: str) -> str:
    if not question or not any(token in question for token in _CONTEXT_REFERENCE_TERMS):
        return ""
    if _has_any_term(last_reply, _NEWS_TERMS):
        return "news"
    if _has_any_term(last_reply, _WEATHER_TERMS):
        return "weather"
    if _has_any_term(last_reply, _ORIGIN_QUESTION_TERMS):
        return "journey"
    return ""


def _should_force_journey(
    question: str,
    question_lc: str,
    last_reply_lc: str,
    last_reply: str | None,
) -> bool:
    if _has_any_term(last_reply_lc, _ORIGIN_QUESTION_TERMS) and extract_origin(question, last_reply):
        return True
    if is_journey_planning_question(question):
        return True
    return _infer_followup_topic(question_lc, last_reply_lc) == "journey"


def _resolve_followup_signals(question_lc: str, last_reply_lc: str) -> tuple[bool, bool]:
    has_weather = _has_any_term(question_lc, _WEATHER_TERMS)
    has_news = _has_any_term(question_lc, _NEWS_TERMS)

    inferred = _infer_followup_topic(question_lc, last_reply_lc)
    if inferred == "news":
        has_news = True
    elif inferred == "weather":
        has_weather = True

    if not has_weather and not has_news:
        if _looks_like_news_followup(question_lc, last_reply_lc):
            has_news = True
        elif _looks_like_weather_followup(question_lc, last_reply_lc):
            has_weather = True

    return has_weather, has_news


def needs_followup_reference_clarification(question: str | None, last_reply: str | None = None) -> bool:
    if not question or last_reply:
        return False

    q = question.lower()
    if not any(re.search(rf"\b{re.escape(term)}\b", q) for term in _CONTEXT_REFERENCE_TERMS):
        return False

    relevance_terms = (
        "what does",
        "what is",
        "how does",
        "how is",
        "why does",
        "why is",
        "affect",
        "impact",
        "matter",
        "has to do with",
    )
    if not any(term in q for term in relevance_terms):
        return False

    concrete_topic_tokens = _meaningful_tokens(question) - set(_CONTEXT_REFERENCE_TERMS)
    return len(concrete_topic_tokens) < 2


def is_journey_planning_question(q: str | None) -> bool:
    if not q:
        return False

    s = q.lower()
    if _has_any_term(s, _JOURNEY_TERMS):
        return True

    if _mentions_transport_choice(s):
        return True

    has_route_phrase = bool(re.search(r"\bfrom\s+\S+", s)) and bool(re.search(r"\bto\s+\S+", s))
    return has_route_phrase and any(token in s for token in ("trip", "travel", "route", "transport", "transpo"))


def asks_route_or_transport(question: str | None) -> bool:
    if not question:
        return False
    q = question.lower()
    return _has_any_term(q, _ROUTE_TRANSPORT_TERMS) or _mentions_transport_choice(q)


def needs_origin_clarification(question: str | None, last_reply: str | None = None) -> bool:
    if not question:
        return False
    last = (last_reply or "").lower()
    if not is_journey_planning_question(question) and not _has_any_term(last, _ORIGIN_QUESTION_TERMS):
        return False
    return extract_origin(question, last_reply) is None


def extract_origin(question: str | None, last_reply: str | None = None) -> str | None:
    if not question:
        return None

    q = " ".join(question.strip().split())
    if not q:
        return None

    patterns = (
        r"\bfrom\s+(.+?)\s+\bto\b",
        r"\bfrom\s+(.+?)(?:[?.!,;]|$)",
        r"\bi'?m in\s+(.+?)(?:[?.!,;]|$)",
        r"\bi am in\s+(.+?)(?:[?.!,;]|$)",
        r"\bi'?m at\s+(.+?)(?:[?.!,;]|$)",
        r"\bcurrently in\s+(.+?)(?:[?.!,;]|$)",
        r"\bbased in\s+(.+?)(?:[?.!,;]|$)",
        r"\blive in\s+(.+?)(?:[?.!,;]|$)",
        r"\bliving in\s+(.+?)(?:[?.!,;]|$)",
        r"\bhere in\s+(.+?)(?:[?.!,;]|$)",
        r"\btraveling from\s+(.+?)(?:\s+\bto\b|$)",
        r"\btravelling from\s+(.+?)(?:\s+\bto\b|$)",
        r"\bcoming from\s+(.+?)(?:\s+\bto\b|$)",
        r"\bdeparting from\s+(.+?)(?:\s+\bto\b|$)",
        r"^\s*from\s+(.+)$",
    )

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if not match:
            continue
        origin = _clean_location_fragment(match.group(1))
        if origin:
            return origin

    if _has_any_term((last_reply or "").lower(), _ORIGIN_QUESTION_TERMS) and _looks_like_location_reply(q):
        return _clean_location_fragment(q)

    return None


def is_origin_only_reply(question: str | None) -> bool:
    if not question:
        return False

    origin = extract_origin(question)
    if not origin:
        return False

    q_lc = question.lower()
    if _has_any_term(q_lc, _WEATHER_TERMS) or _has_any_term(q_lc, _NEWS_TERMS):
        return False
    if _has_any_term(q_lc, _TRAVEL_BRIEF_TERMS):
        return False
    if is_trip_planning_question(question) or is_journey_planning_question(question):
        return False
    if asks_route_or_transport(question):
        return False

    compact = " ".join(question.strip().split())
    return _looks_like_location_reply(compact) or len(compact.split()) <= 6


def _clean_location_fragment(fragment: str) -> str | None:
    cleaned = re.split(r"[?.!,;]| by | via | using ", fragment, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    cleaned = re.sub(r"^(the)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def _looks_like_location_reply(text: str) -> bool:
    compact = " ".join((text or "").split())
    if not compact or "?" in compact:
        return False
    if len(compact.split()) > 5:
        return False
    if re.search(
        r"\b(bus|car|drive|flight|ferry|route|transport|transpo|weather|news|risk|safe|should|can|what|when|why|how)\b",
        compact,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", compact))


def _mentions_transport_choice(text: str) -> bool:
    if not text:
        return False

    has_transport_mode = any(re.search(rf"\b{re.escape(term)}\b", text) for term in _TRANSPORT_MODE_TERMS)
    if not has_transport_mode:
        return False

    if _has_any_term(text, _ROUTE_TRANSPORT_TERMS):
        return True

    return " or " in text


def _token_overlap(a: str, b: str) -> set[str]:
    a_tokens = _meaningful_tokens(a)
    b_tokens = _meaningful_tokens(b)
    return a_tokens & b_tokens


def _meaningful_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{4,}", text.lower()) if token not in _STOPWORDS}


def _looks_like_news_followup(question: str, last_reply: str) -> bool:
    if not last_reply:
        return False
    last_lc = last_reply.lower()
    if "recent local reporting" in last_lc or "retrieved news" in last_lc or _has_any_term(last_lc, _NEWS_TERMS):
        if _token_overlap(question, last_reply):
            return True
        if _has_any_term(question, _FOLLOWUP_CONTINUATION_TERMS):
            return True
    return False


def _looks_like_weather_followup(question: str, last_reply: str) -> bool:
    if not last_reply:
        return False
    last_lc = last_reply.lower()
    if _has_any_term(last_lc, _WEATHER_TERMS):
        if _token_overlap(question, last_reply):
            return True
        if _has_any_term(question, _FOLLOWUP_CONTINUATION_TERMS):
            return True
    return False
