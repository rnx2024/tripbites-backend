import unittest

from app.news.news_relevance import (
    contains_high_impact_claim,
    filter_relevant_news,
    news_item_mentions_place,
    sanitize_answer_links,
    supports_high_impact_claim,
)


class NewsRelevanceTests(unittest.TestCase):
    def test_vigan_item_is_relevant(self) -> None:
        item = {"title": "Classes and work suspended in Vigan City", "snippet": "An executive order applies today."}
        self.assertTrue(news_item_mentions_place(item, "Vigan"))

    def test_rizal_item_is_rejected_for_vigan(self) -> None:
        item = {"title": "Authorities clear debris on highway in Rizal", "snippet": "A landslide blocked the road."}
        self.assertFalse(news_item_mentions_place(item, "Vigan"))
        self.assertEqual(filter_relevant_news([item], "Vigan"), [])

    def test_high_impact_claim_requires_matching_place_and_topic(self) -> None:
        answer = "Classes and work are suspended in Vigan City today."
        supported = {
            "title": "Classes and work suspended in Vigan City",
            "snippet": "The suspension follows an executive order.",
        }
        unrelated = {"title": "Authorities clear debris on highway in Rizal", "snippet": "Landslide reported."}

        self.assertTrue(contains_high_impact_claim(answer))
        self.assertTrue(supports_high_impact_claim(answer, supported, "Vigan"))
        self.assertFalse(supports_high_impact_claim(answer, unrelated, "Vigan"))

    def test_sanitize_answer_links_removes_unvalidated_urls(self) -> None:
        answer = "Read [the update](https://example.com/vigan) and https://example.com/rizal."
        self.assertEqual(
            sanitize_answer_links(answer, {"https://example.com/vigan"}),
            "Read [the update](https://example.com/vigan) and ",
        )


if __name__ == "__main__":
    unittest.main()
