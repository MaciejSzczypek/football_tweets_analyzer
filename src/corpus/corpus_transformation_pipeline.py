from typing import List

from nptyping import Array

from configs.config_schema import HyperParametersConfig
from corpus.cleaning.corpus_normalizer import TokenizedCorpusNormalizer
from src.corpus.tokenizing.corpus_tokenizer import CorpusTokenizer
from dataclasses import dataclass


@dataclass()
class TransformedCorpus:
    corpus: List[List[str]]
    indexes_of_removed_tweets: List[int]


class CorpusTransformer:
    @classmethod
    def transform_twitter_corpus(
            cls,
            corpus: Array[str],
            config: HyperParametersConfig,
            remove_empty: bool = True,
    ) -> TransformedCorpus:
        tokenized_tweets = CorpusTokenizer.tokenize(
            corpus=corpus,
            word_tokenizer=config.corpus.tokenization.word_tokenizer,
            sentence_tokenizer=config.corpus.tokenization.sentence_tokenizer,
        )

        tokenized_corpus_normalizer = TokenizedCorpusNormalizer()
        normalized_tweets = tokenized_corpus_normalizer.normalize(
            tokenized_corpus=tokenized_tweets,
            normalization_operations_config=config.corpus.normalization.operations,
        )

        if remove_empty:
            normalized_tweets, removed_indexes = cls._filter_empty_tweets(normalized_tweets)
        else:
            removed_indexes = []

        return TransformedCorpus(
            corpus=normalized_tweets,
            indexes_of_removed_tweets=removed_indexes,
        )

    @staticmethod
    def _filter_empty_tweets(tweets: List[List[str]]) -> (List[List[str]], List[int]):
        filtered_tweets = [tweet for tweet in tweets if tweet]
        removed_indexes = [i for i, tweet in enumerate(tweets) if not tweet]
        return filtered_tweets, removed_indexes

