import unittest
from corpus.cleaning.constants import FOOTBALL_RESULT_FORMAT_REGEX
from corpus.tokenizing.custom_tokenizers import CustomTokenizer


class TestCustomTokenizer(unittest.TestCase):

    def test_tokenize_with_normal_text(self):
        text = "This is a test sentence."
        expected_result = ["This", "is", "a", "test", "sentence."]
        self.assertEqual(CustomTokenizer.tokenize(text), expected_result)

    def test_tokenize_with_football_result(self):
        text = "The match ended with a score of 2:1 after a thrilling game."
        expected_result = ["The", "match", "ended", "with", "a", "score", "of", "2-1", "after", "a", "thrilling",
                           "game."]
        self.assertEqual(CustomTokenizer.tokenize(text), expected_result)

    def test_normalize_football_result_in_text(self):
        text = "The final score was 3:2 in favor of the home team."
        expected_result = "The final score was 3-2 in favor of the home team."
        self.assertEqual(CustomTokenizer._normalize_football_match_result_if_occurs(text), expected_result)

    def test_tokenize_with_no_football_result(self):
        text = "There is no football match score here."
        expected_result = ["There", "is", "no", "football", "match", "score", "here."]
        self.assertEqual(CustomTokenizer.tokenize(text), expected_result)


if __name__ == "__main__":
    unittest.main()
