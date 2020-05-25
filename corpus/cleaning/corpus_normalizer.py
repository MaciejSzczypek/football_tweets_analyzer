from typing import List, Optional, Set

from corpus.cleaning.tweet_text_normalizer import TweetTextNormalizer
from configs.config_schema import OperationsForTextNormalizationConfig


class TokenizedCorpusNormalizer:
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self._text_normalizer = TweetTextNormalizer(stopwords=stopwords)

    def normalize(
        self,
        tokenized_corpus: List[List[str]],
        normalization_operations_config: OperationsForTextNormalizationConfig,
    ):
        return [
            self._text_normalizer.normalize(
                tokenized_text=document,
                **normalization_operations_config.__dict__,
            )
            for document in tokenized_corpus
        ]
