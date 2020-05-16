from typing import List

import numpy as np
from nptyping import Array
from configs.config_schema import Config
from corpus.cleaning.corpus_normalizer import TokenizedCorpusNormalizer
from corpus.cleaning.tweet_text_extractor import TweetTextExtractor
from corpus.tokenizing.corpus_tokenizer import CorpusTokenizer


class CorpusTransformationPipeline:
    @classmethod
    def transform_twitter_corpus(
        cls, corpus: Array[str], config: Config,
    ) -> List[List[str]]:
        vectorized_tweet_text_extractor = np.vectorize(
            TweetTextExtractor.extract_text_from_tweet
        )
        extracted_tweets = vectorized_tweet_text_extractor(corpus)

        tokenized_tweets = CorpusTokenizer.tokenize(extracted_tweets)

        tokenized_corpus_normalizer = TokenizedCorpusNormalizer()
        tokenized_and_normalized_tweets = tokenized_corpus_normalizer.normalize(
            tokenized_corpus=tokenized_tweets,
            remove_stopwords=config.settings[0].corpus.text_normalization.operations.remove_stopwords,
        )
        return tokenized_and_normalized_tweets
