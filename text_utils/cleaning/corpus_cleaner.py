import numpy as np
from nptyping import  Array
from text_utils.text_standardizer import TextNormalizer
from text_utils.cleaning.tweet_cleaner import TweetCleaner


class CorpusCleaner:

    @classmethod
    def clean_text_corpus(cls, corpus: Array[str]) -> Array[str]:
        vectorized_tweet_cleaner = np.vectorize(TweetCleaner.clean_text_from_tweet)
        corpus = vectorized_tweet_cleaner(corpus)
        vectorized_text_normalizer = np.vectorize(TextNormalizer.normalize)
        corpus = vectorized_text_normalizer(corpus)
        return corpus

