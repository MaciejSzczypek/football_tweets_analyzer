import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from dataclasses import dataclass
from feature_extraction.document_similarity import get_cosine_similarity_df_from_tfidf_matrix
from pprint import pprint
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME


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
    def generate_tweets_summary(cls, tweets, top_n_grams, threshold):
        # count score based on ngrams approach!!!
        sentence_count = 0

        n_grams = [ngram_text for ngram_text, ngram_count in top_n_grams]
        summary_sentences = []
        for tweet in tweets:
            tweet = " ".join(tweet)
            for n_gram in n_grams:
                if n_gram in tweet:
                    summary_sentences.append(tweet)

        # for sentence in sentences:
        #     if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
        #         summary += " " + sentence
        #         sentence_count += 1
        summary = ".".join(summary_sentences)
        return summary

    @classmethod
    def generate_tweets_summary_based_on_page_rank_and_random_sample(
            cls,
            tfidf_tweets: pd.DataFrame,
            df_before_transformation: pd.DataFrame,
            sentences: List[List[str]],
            top_n_tweets: int = 10,
            random_batch_size: int = 28000,
    ):
        random_indexes = tfidf_tweets.sample(n=random_batch_size, random_state=22).index
        tfidf_tweets_random_batch = tfidf_tweets[tfidf_tweets.index.isin(random_indexes)]
        sentences_from_random_batch = [
            SentenceInfoKeeper(sentence=sentence, initial_index=index)
            for index, sentence in enumerate(sentences)
            if index in random_indexes
        ]
        similarity_matrix = get_cosine_similarity_df_from_tfidf_matrix(tfidf_tweets_random_batch)
        sentence_similarity_graph = nx.from_numpy_array(similarity_matrix.to_numpy())
        scores = nx.pagerank(sentence_similarity_graph)
        ranked_sentence = sorted(
            (
                ScoreInfoKeeper(
                    score=scores[index],
                    initial_index=sentence_info_keeper.initial_index,
                    sentence=sentence_info_keeper.sentence,
                )
                for index, sentence_info_keeper
                in enumerate(sentences_from_random_batch)
            ),
            key=lambda score_info: score_info.score,
            reverse=True
        )

        top_tweets = {
            cls.NORMALIZED_SENTENCE_COLUMN_NAME: [],
            cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME: [],
        }
        for index in range(top_n_tweets):
            normalized_sentence = " ".join(ranked_sentence[index].sentence)
            normalized_sentence_initial_index = ranked_sentence[index].initial_index
            pre_normalized_sentence = (
                df_before_transformation
                .iloc
                [normalized_sentence_initial_index]
                [LIV_WAT_TEXT_COLUMN_NAME]
            )

            top_tweets[cls.NORMALIZED_SENTENCE_COLUMN_NAME].append(normalized_sentence)
            top_tweets[cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME].append(pre_normalized_sentence)

        top_tweets_df = pd.DataFrame.from_dict(top_tweets)

        print(f"Top {top_n_tweets} of the most relevant tweets:")
        for index, tweet in top_tweets_df.iterrows():
            print(f"{index + 1}. {tweet[cls.PRE_NORMALIZED_SENTENCE_COLUMN_NAME]}")
