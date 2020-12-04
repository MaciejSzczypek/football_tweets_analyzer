from dataclasses import dataclass
from typing import List

import networkx as nx
import pandas as pd

from data.column_names import TEXT_COLUMN_NAME
from feature_extraction.document_similarity import (
    get_cosine_similarity_df_from_tfidf_matrix,
)


@dataclass
class SentenceInfoKeeper:
    sentence: List[str]
    initial_index: int


@dataclass
class ScoreInfoKeeper:
    score: float
    initial_index: int
    sentence: List[str]


class TweetsSummarizer:
    NORMALIZED_SENTENCE_COLUMN_NAME = "normalized_sentence"
    PRE_NORMALIZED_SENTENCE_COLUMN_NAME = "pre_normalized_sentence"

    @classmethod
    def generate_tweets_summary_based_on_page_rank_and_random_sample(
        cls,
        tfidf_tweets: pd.DataFrame,
        df_before_transformation: pd.DataFrame,
        sentences: List[List[str]],
        top_n_tweets: int = 10,
        random_batch_size: int = 28000,
    ):
        random_indexes = tfidf_tweets.sample(n=random_batch_size, random_state=1).index
        tfidf_tweets_random_batch = tfidf_tweets[
            tfidf_tweets.index.isin(random_indexes)
        ]
        sentences_from_random_batch = [
            SentenceInfoKeeper(sentence=sentence, initial_index=index)
            for index, sentence in enumerate(sentences)
            if index in random_indexes
        ]
        similarity_matrix = get_cosine_similarity_df_from_tfidf_matrix(
            tfidf_tweets_random_batch
        )
        sentence_similarity_graph = nx.from_numpy_array(similarity_matrix.to_numpy())
        scores = nx.pagerank(sentence_similarity_graph)
        ranked_sentence = sorted(
            (
                ScoreInfoKeeper(
                    score=scores[index],
                    initial_index=sentence_info_keeper.initial_index,
                    sentence=sentence_info_keeper.sentence,
                )
                for index, sentence_info_keeper in enumerate(
                    sentences_from_random_batch
                )
            ),
            key=lambda score_info: score_info.score,
            reverse=True,
        )

        top_tweets = {
            cls.NORMALIZED_SENTENCE_COLUMN_NAME: [],
            cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME: [],
        }
        for index in range(top_n_tweets):
            normalized_sentence = " ".join(ranked_sentence[index].sentence)
            normalized_sentence_initial_index = ranked_sentence[index].initial_index
            pre_normalized_sentence = df_before_transformation.iloc[
                normalized_sentence_initial_index
            ][TEXT_COLUMN_NAME]

            top_tweets[cls.NORMALIZED_SENTENCE_COLUMN_NAME].append(normalized_sentence)
            top_tweets[cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME].append(
                pre_normalized_sentence
            )

        top_tweets_df = pd.DataFrame.from_dict(top_tweets)

        print(f"Top {top_n_tweets} of the most relevant tweets:")
        for index, tweet in top_tweets_df.iterrows():
            print(f"{index + 1}. {tweet[cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME]}")
