import unittest
from unittest.mock import patch


from corpus.cleaning.tweet_text_normalizer import TweetTextNormalizer, CONTRACTION_EXTENSIONS

class TestTweetTextNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = TweetTextNormalizer(stopwords={"a", "the", "in"})

    def test_stopwords_initialization(self):
        self.assertEqual(self.normalizer.stopwords, {"a", "the", "in"})

    def test_normalize_lower(self):
        tokenized_text = ["Hello", "WORLD", "This", "Is", "A", "TEST"]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=True, remove_commas=False, remove_stopwords=False,
            apply_stemming=False, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, ["hello", "world", "this", "is", "a", "test"])

    def test_remove_commas(self):
        tokenized_text = ["Hello,", "world,", "this", "is", "a", "test."]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=True, remove_stopwords=False,
            apply_stemming=False, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, ["Hello", "world", "this", "is", "a", "test."])

    def test_remove_stopwords(self):
        tokenized_text = ["This", "is", "a", "simple", "test"]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=False, remove_stopwords=True,
            apply_stemming=False, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, ["This", "is", "simple", "test"])

    def test_apply_stemming(self):
        tokenized_text = ["running", "jumps", "flies"]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=False, remove_stopwords=False,
            apply_stemming=True, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, ["run", "jump", "fli"])

    def test_apply_lemmatization(self):
        tokenized_text = ["running", "jumps", "flies"]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=False, remove_stopwords=False,
            apply_stemming=False, apply_lemmatization=True, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, ["running", "jump", "fly"])

    def test_expand_contractions(self):
        with patch.dict(CONTRACTION_EXTENSIONS, {"don't": "do not", "isn't": "is not"}):
            tokenized_text = ["don't", "worry", "isn't", "this", "a", "test"]
            normalized_text = self.normalizer.normalize(
                tokenized_text, lower=False, remove_commas=False, remove_stopwords=False,
                apply_stemming=False, apply_lemmatization=False, expand_contractions=True,
                remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
                remove_user_mentions=False, remove_links=False
            )
            self.assertEqual(normalized_text, ["do", "not", "worry", "is", "not", "this", "a", "test"])

    def test_remove_special_characters(self):
        tokenized_text = ["hello!", "world&", "this*", "is^", "a", "test$"]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=False, remove_stopwords=False,
            apply_stemming=False, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=True, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, ["hello", "world", "this", "is", "a", "test"])

    def test_empty_tokenized_text(self):
        tokenized_text = []
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=False, remove_stopwords=False,
            apply_stemming=False, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, [])

    def test_no_normalization(self):
        tokenized_text = ["Hello", "WORLD", "This", "Is", "A", "TEST"]
        normalized_text = self.normalizer.normalize(
            tokenized_text, lower=False, remove_commas=False, remove_stopwords=False,
            apply_stemming=False, apply_lemmatization=False, expand_contractions=False,
            remove_special_characters=False, remove_emoticons=False, remove_hashtags=False,
            remove_user_mentions=False, remove_links=False
        )
        self.assertEqual(normalized_text, tokenized_text)

if __name__ == '__main__':
    unittest.main()
