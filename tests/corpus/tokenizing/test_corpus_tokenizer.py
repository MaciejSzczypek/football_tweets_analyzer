import unittest
from unittest.mock import MagicMock
from corpus.tokenizing.corpus_tokenizer import CorpusTokenizer


class TestCorpusTokenizer(unittest.TestCase):

    def setUp(self):
        self.mock_sentence_tokenizer = MagicMock()
        self.mock_word_tokenizer = MagicMock()

    def test_tokenize_with_both_sentence_and_word_tokenizer(self):
        self.mock_sentence_tokenizer.return_value = ["This is a sentence.", "This is another sentence."]
        self.mock_word_tokenizer.return_value = ["This", "is", "a", "sentence"]

        corpus = ["This is a sentence. This is another sentence."]

        result = CorpusTokenizer.tokenize(corpus, self.mock_sentence_tokenizer, self.mock_word_tokenizer)

        self.assertEqual(result, [["This", "is", "a", "sentence", "This", "is", "another", "sentence"]])

        self.mock_sentence_tokenizer.assert_called_once_with(text="This is a sentence. This is another sentence.")

        self.mock_word_tokenizer.assert_any_call(text="This is a sentence.")
        self.mock_word_tokenizer.assert_any_call(text="This is another sentence.")

    def test_tokenize_with_only_word_tokenizer(self):
        self.mock_word_tokenizer.return_value = ["This", "is", "a", "test"]

        corpus = ["This is a test"]

        result = CorpusTokenizer.tokenize(corpus, None, self.mock_word_tokenizer)

        self.assertEqual(result, [["This", "is", "a", "test"]])

        self.mock_word_tokenizer.assert_called_once_with(text="This is a test")

    def test_empty_corpus(self):
        corpus = []
        result = CorpusTokenizer.tokenize(corpus, self.mock_sentence_tokenizer, self.mock_word_tokenizer)

        self.assertEqual(result, [])

        self.mock_sentence_tokenizer.assert_not_called()
        self.mock_word_tokenizer.assert_not_called()

    def test_tokenize_with_empty_document(self):
        self.mock_word_tokenizer.return_value = []

        corpus = [""]
        result = CorpusTokenizer.tokenize(corpus, None, self.mock_word_tokenizer)

        self.assertEqual(result, [[]])

        self.mock_word_tokenizer.assert_called_once_with(text="")

    def test_sentence_tokenizer_is_optional(self):
        self.mock_word_tokenizer.return_value = ["This", "is", "a", "test"]

        corpus = ["This is a test"]

        result = CorpusTokenizer.tokenize(corpus, None, self.mock_word_tokenizer)

        self.assertEqual(result, [["This", "is", "a", "test"]])

        self.mock_word_tokenizer.assert_called_once_with(text="This is a test")

    def test_sentence_and_word_tokenizer_called_in_order(self):
        self.mock_sentence_tokenizer.return_value = ["This is a sentence."]
        self.mock_word_tokenizer.return_value = ["This", "is", "a", "sentence"]

        corpus = ["This is a sentence."]

        result = CorpusTokenizer.tokenize(corpus, self.mock_sentence_tokenizer, self.mock_word_tokenizer)

        self.assertEqual(result, [["This", "is", "a", "sentence"]])

        self.mock_sentence_tokenizer.assert_called_once_with(text="This is a sentence.")
        self.mock_word_tokenizer.assert_called_once_with(text="This is a sentence.")

