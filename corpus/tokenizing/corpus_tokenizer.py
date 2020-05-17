from typing import Callable, List

from nptyping import Array


class CorpusTokenizer:
    @classmethod
    def tokenize(
        cls,
        corpus: Array[str],
        sentence_tokenizer: Callable,
        word_tokenizer: Callable,
    ) -> List[List[str]]:
        tokenized_corpus = [
            cls._tokenize_with_both_sentence_and_word_tokenizer(
                document=document,
                sentence_tokenizer=sentence_tokenizer,
                word_tokenizer=word_tokenizer,
            )
            if sentence_tokenizer
            else cls._tokenize_only_with_word_tokenizer(
                document=document, word_tokenizer=word_tokenizer,
            )
            for document in corpus
        ]

        return tokenized_corpus

    @classmethod
    def _tokenize_with_both_sentence_and_word_tokenizer(
        cls, document: str, sentence_tokenizer: Callable, word_tokenizer: Callable,
    ) -> List[str]:
        sentence_tokenized_document = sentence_tokenizer(text=document)
        word_tokenized_sentences = []
        for sentence in sentence_tokenized_document:
            word_tokenized_sentences.extend(word_tokenizer(text=sentence))
        return word_tokenized_sentences

    @classmethod
    def _tokenize_only_with_word_tokenizer(
        cls, document: str, word_tokenizer: Callable
    ) -> List[str]:
        return word_tokenizer(text=document)
