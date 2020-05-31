from typing import List

from nptyping import Array

from configs.config_schema import HyperParametersConfig
from corpus.cleaning.corpus_normalizer import TokenizedCorpusNormalizer
from corpus.tokenizing.corpus_tokenizer import CorpusTokenizer


class CorpusTransformer:
    @classmethod
    def transform_twitter_corpus(
        cls,
        corpus: Array[str],
        hyper_parameters_config: HyperParametersConfig,
        remove_empty_tweets: bool = True,
    ) -> List[List[str]]:
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
        if remove_empty_tweets:
            tokenized_and_normalized_tweets = [
                tweet for tweet in tokenized_and_normalized_tweets if tweet
            ]
        return tokenized_and_normalized_tweets
