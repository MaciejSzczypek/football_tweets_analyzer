from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from configs.config_schema import HyperParametersConfig
from corpus.corpus_transformation_pipeline import CorpusTransformer
from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME, CREATED_AT_COLUMN_NAME
from emoji import emoji_count
from emoji.unicode_codes import UNICODE_EMOJI

import demoji


@dataclass(frozen=True)
class DataSet:
    initial_df: pd.DataFrame
    normalized_tweets_as_token_lists: List[List[str]]
    tfidf_vectorizer: TfidfVectorizer
    original_df: Optional[pd.DataFrame]

    # todo transform to lazy properties
    @property
    def normalized_tweets_as_strings(self) -> List[str]:
        return [" ".join(tweet) for tweet in self.normalized_tweets_as_token_lists]

    @property
    def normalized_tweets_as_demojized_strings(self) -> List[str]:
        return [
            " ".join(self._convert_tokens_with_emojis_to_text(tweet))
            for tweet in self.normalized_tweets_as_token_lists
        ]

    @property
    def initial_df_with_emojis_converted_to_text(self) -> pd.DataFrame:
        df = self.initial_df.copy()
        df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].apply(self._convert_emojis_to_text)
        return df

    @property
    def initial_tweets_array(self) -> List[str]:
        return self.initial_df[TEXT_COLUMN_NAME].to_numpy()

    @property
    def original_tweets_array(self) -> Optional[List[str]]:
        print(self.initial_df["Unnamed: 0.1.1"])
        if self.original_df is not None:
            return self.original_df.iloc[self.initial_df["Unnamed: 0.1.1"]][TEXT_COLUMN_NAME].to_numpy()

    @property
    def df_with_normalized_tweets(self) -> pd.DataFrame:
        df = self.initial_df.copy()
        df[TEXT_COLUMN_NAME] = self.normalized_tweets_as_strings
        return df

    @property
    def flat_text(self) -> str:
        return ". ".join(self.normalized_tweets_as_strings)

    def prepare_for_nltk_classifier(
            self, convert_emojis_to_text: bool = False
    ) -> List[Tuple[Dict[str, bool], float]]:
        df = self.initial_df.copy()
        df[TEXT_COLUMN_NAME] = self.normalized_tweets_as_token_lists
        if convert_emojis_to_text:
            df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].apply(
                self._convert_tokens_with_emojis_to_text
            )
        return list(
            df.apply(
                self._prepare_nltk_training_record, axis=1
            )
        )

    @property
    def tfidf(self):
        return self.tfidf_vectorizer.fit_transform(self.normalized_tweets_as_strings)

    @property
    def tfidf_for_aggregated_tweets(self):
        aggregated_tweets = self._get_aggregated_tweets()
        return self.tfidf_vectorizer.fit_transform(aggregated_tweets)

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        hyper_parameters_config: HyperParametersConfig,
        tfidf_vectorizer=None,
        original_df: Optional[pd.DataFrame] = None
    ) -> "DataSet":
        tweets_array = df[TEXT_COLUMN_NAME].to_numpy()
        transformed_corpus = CorpusTransformer.transform_twitter_corpus(
            corpus=tweets_array,
            hyper_parameters_config=hyper_parameters_config,
        )
        df = cls._remove_df_rows(
            df=df,
            rows_to_remove_indexes=transformed_corpus.indexes_of_removed_tweets,
        )
        if not tfidf_vectorizer:
            tfidf_vectorizer = TfidfVectorizer(
                token_pattern=r"\S+", stop_words="english", min_df=20, max_features=1350,
            )
        return cls(
            initial_df=df,
            normalized_tweets_as_token_lists=transformed_corpus.corpus,
            tfidf_vectorizer=tfidf_vectorizer,
            original_df=original_df,
        )

    def _get_aggregated_tweets(
        self, aggregation_factor: int = 5,
    ) -> List[str]:
        aggregated_tweets = []
        counter = 0
        while True:
            start = counter * aggregation_factor
            stop = (counter + 1) * aggregation_factor
            if stop < len(self.normalized_tweets_as_strings):
                aggregated_tweets.append(" ".join(self.normalized_tweets_as_strings[start:stop]))
            else:
                aggregated_tweets.append(" ".join(self.normalized_tweets_as_strings[start:]))
                break
            counter += 1
        return aggregated_tweets

    @classmethod
    def _remove_df_rows(cls, df: pd.DataFrame, rows_to_remove_indexes):
        return df[~df.index.isin(df.iloc[rows_to_remove_indexes].index)]

    @classmethod
    def _prepare_nltk_training_record(cls, record: pd.Series) -> Tuple[Dict[str, bool], float]:
        tokens_dict = {token: True for token in record[TEXT_COLUMN_NAME]}
        return tokens_dict, record[LABEL_COLUMN_NAME]

    @classmethod
    def _convert_tokens_with_emojis_to_text(cls, tokens: List[str]) -> List[str]:
        return [
            cls._convert_emojis_to_text(token) if emoji_count(token) else token
            for token in tokens
        ]

    @classmethod
    def _convert_emojis_to_text(cls, text: str) -> str:
        emojis_map = demoji.findall(text)
        text_with_spaces_before_emojis = ""
        for character in text:
            if character in UNICODE_EMOJI:
                text_with_spaces_before_emojis += " "
            text_with_spaces_before_emojis += character
        text_with_emojis_converted_to_text = text_with_spaces_before_emojis
        for emoji, emoji_text_value in emojis_map.items():
            text_with_emojis_converted_to_text = (
                text_with_emojis_converted_to_text.replace(emoji, emoji_text_value)
            )
        return text_with_emojis_converted_to_text



