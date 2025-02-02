from dataclasses import dataclass
from typing import List

import networkx as nx
import pandas as pd

from feature_extraction.document_similarity import get_cosine_similarity_df_from_tfidf_matrix
from feature_extraction.tfidf import create_df_with_tfidf_feature_vectors
from utils.printing import section_printing_decorator


@dataclass
class SentenceInfo:
    sentence: List[str]
    initial_index: int


@dataclass
class ScoreInfo:
    score: float
    initial_index: int
    sentence: List[str]

class TweetsSummarizer:
    NORMALIZED_SENTENCE_COLUMN_NAME = "normalized_sentence"
    PRE_NORMALIZED_SENTENCE_COLUMN_NAME = "pre_normalized_sentence"
    TEXT_COLUMN_NAME = "text"
    DEFAULT_TOP_TWEETS = 10
    DEFAULT_RANDOM_STATE = 1

    @classmethod
    def generate_most_relevant_sentences(
        cls,
        tfidf_tweets: pd.DataFrame,
        df_before_transformation: pd.DataFrame,
        sentences: List[List[str]],
        random_batch_size: int,
        top_n_tweets: int = DEFAULT_TOP_TWEETS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> pd.Series:
        random_indexes, tfidf_tweets_random_batch = cls._get_random_batch(
            tfidf_tweets, random_batch_size, random_state
        )
        sentences_from_random_batch = cls._extract_sentences_from_batch(sentences, random_indexes)
        ranked_sentences = cls._compute_sentence_scores(tfidf_tweets_random_batch, sentences_from_random_batch)

        top_tweets_data = cls._prepare_top_tweets_data(
            ranked_sentences, df_before_transformation, top_n_tweets
        )

        return pd.DataFrame.from_dict(top_tweets_data)[cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME]

    @staticmethod
    def _get_random_batch(
        tfidf_tweets: pd.DataFrame, batch_size: int, random_state: int
    ) -> tuple:
        random_indexes = tfidf_tweets.sample(n=batch_size, random_state=random_state).index
        tfidf_tweets_random_batch = tfidf_tweets[tfidf_tweets.index.isin(random_indexes)]
        return random_indexes, tfidf_tweets_random_batch

    @staticmethod
    def _extract_sentences_from_batch(sentences: List[List[str]], random_indexes: pd.Index) -> List[SentenceInfo]:
        return [
            SentenceInfo(sentence=sentence, initial_index=index)
            for index, sentence in enumerate(sentences)
            if index in random_indexes
        ]

    @staticmethod
    def _compute_sentence_scores(
        tfidf_tweets_random_batch: pd.DataFrame, sentences_from_random_batch: List[SentenceInfo]
    ) -> List[ScoreInfo]:
        similarity_matrix = get_cosine_similarity_df_from_tfidf_matrix(tfidf_tweets_random_batch)
        sentence_similarity_graph = nx.from_numpy_array(similarity_matrix.to_numpy())
        scores = nx.pagerank(sentence_similarity_graph)
        return sorted(
            (
                ScoreInfo(
                    score=scores[index],
                    initial_index=sentence_info_keeper.initial_index,
                    sentence=sentence_info_keeper.sentence,
                )
                for index, sentence_info_keeper in enumerate(sentences_from_random_batch)
            ),
            key=lambda score_info: score_info.score,
            reverse=True,
        )

    @classmethod
    def _prepare_top_tweets_data(
        cls,
        ranked_sentences: List[ScoreInfo],
        df_before_transformation: pd.DataFrame,
        top_n_tweets: int,
    ) -> dict:
        top_tweets_data = {
            cls.NORMALIZED_SENTENCE_COLUMN_NAME: [],
            cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME: [],
        }
        for index in range(min(top_n_tweets, len(ranked_sentences))):
            normalized_sentence = " ".join(ranked_sentences[index].sentence)
            normalized_sentence_initial_index = ranked_sentences[index].initial_index
            pre_normalized_sentence = df_before_transformation.iloc[
                normalized_sentence_initial_index
            ][cls.TEXT_COLUMN_NAME]

            top_tweets_data[cls.NORMALIZED_SENTENCE_COLUMN_NAME].append(normalized_sentence)
            top_tweets_data[cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME].append(pre_normalized_sentence)

        return top_tweets_data


@section_printing_decorator("MOST RELEVANT TWEETS")
def print_most_relevant_sentences(
    normalized_tweets_as_strings: List[str],
    df_before_transformation: pd.DataFrame,
    sentences: List[List[str]],
    maximum_number_of_features: int = 100,
    random_batch_size: int = None,
):
    if not random_batch_size:
        random_batch_size = len(sentences)
    tfidf_df = create_df_with_tfidf_feature_vectors(
        corpus=normalized_tweets_as_strings, maximum_number_of_features=maximum_number_of_features,
    )
    sentences = TweetsSummarizer.generate_most_relevant_sentences(
        tfidf_tweets=tfidf_df,
        df_before_transformation=df_before_transformation,
        sentences=sentences,
        random_batch_size=random_batch_size
    )
    for i, row in enumerate(sentences):
        print(f"{i + 1}. {row}")
