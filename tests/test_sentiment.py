"""Unit tests for Twitter sentiment analysis."""
import unittest

class TestTextPreprocessing(unittest.TestCase):
    def test_remove_urls(self):
        import re
        text = "Check this out https://example.com great stuff"
        cleaned = re.sub(r'https?://\S+', '', text).strip()
        self.assertNotIn("https://", cleaned)

    def test_remove_mentions(self):
        import re
        text = "@user1 hello @user2 how are you"
        cleaned = re.sub(r'@\w+', '', text).strip()
        self.assertNotIn("@", cleaned)

    def test_remove_hashtags(self):
        import re
        text = "Loving #Python and #DataScience today"
        cleaned = re.sub(r'#\w+', '', text).strip()
        self.assertNotIn("#", cleaned)

    def test_lowercase(self):
        self.assertEqual("HELLO WORLD".lower(), "hello world")

    def test_empty_after_cleaning(self):
        import re
        text = "@user #tag https://url.com"
        cleaned = re.sub(r'(@\w+|#\w+|https?://\S+)', '', text).strip()
        self.assertEqual(cleaned, "")

class TestSentimentLabels(unittest.TestCase):
    def test_positive_score_high(self):
        scores = {"positive": 0.91, "neutral": 0.06, "negative": 0.03}
        self.assertEqual(max(scores, key=scores.get), "positive")

    def test_negative_score_high(self):
        scores = {"positive": 0.04, "neutral": 0.08, "negative": 0.88}
        self.assertEqual(max(scores, key=scores.get), "negative")

if __name__ == "__main__": unittest.main()
