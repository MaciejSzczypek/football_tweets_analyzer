from typing import List

import numpy as np
from nptyping import Array

from configs.config_schema import HyperParametersConfig
from corpus.cleaning.corpus_normalizer import TokenizedCorpusNormalizer
from corpus.cleaning.tweet_text_extractor import TweetTextExtractor
from corpus.tokenizing.corpus_tokenizer import CorpusTokenizer


class CorpusTransformer:
    @classmethod
    def transform_twitter_corpus(
        cls, corpus: Array[str], hyper_parameters_config: HyperParametersConfig,
    ) -> List[List[str]]:
        # vectorized_tweet_text_extractor = np.vectorize(
        #     TweetTextExtractor.extract_text_from_tweet
        # )
        # extracted_tweets = vectorized_tweet_text_extractor(corpus)

        tokenized_tweets = CorpusTokenizer.tokenize(
            corpus=corpus,
            word_tokenizer=hyper_parameters_config.corpus.tokenization.word_tokenizer,
            sentence_tokenizer=hyper_parameters_config.corpus.tokenization.sentence_tokenizer,
        )

        tokenized_corpus_normalizer = TokenizedCorpusNormalizer()

        normalization_operations_config = (
            hyper_parameters_config.corpus.normalization.operations
        )
        tokenized_and_normalized_tweets = tokenized_corpus_normalizer.normalize(
            tokenized_corpus=tokenized_tweets,
            normalization_operations_config=normalization_operations_config,
        )

        tokenized_and_normalized_tweets = [
            tweet for tweet in tokenized_and_normalized_tweets if tweet
        ]
        # tokenized_and_normalized_tweets = [
        #     " ".join(tweet) for tweet in tokenized_and_normalized_tweets if tweet
        # ]     # for tfidf_vectorizer
        return tokenized_and_normalized_tweets
