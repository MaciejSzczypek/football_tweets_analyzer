from gensim.summarization.summarizer import summarize, summarize_corpus
from gensim.summarization import keywords, mz_keywords
import matplotlib.pyplot as plt
from configs.config_loader import ConfigLoader
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover
from corpus.corpus_transformation_pipeline import CorpusTransformer
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from data.loaders import DataLoader
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora
import random
import transformers
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint
from sklearn.preprocessing import normalize
from pytorch_transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import tomotopy as tp
from data.paths import (
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_PATH,
)
from information_extraction.keyphrase_extraction import get_top_ngrams
from information_extraction.text_summarization import TweetsSummarizer
from feature_extraction.tfidf import create_df_with_tfidf_feature_vectors
from information_extraction.facts_extractor import FactsExtractor
from information_extraction.keyphrase_extraction import get_tfidf_weighted_keyphrases
from typing import Callable, List
from feature_extraction.tweet_scoring import score_tweets
from feature_extraction.word_frequency import count_word_occurences
import pandas as pd
import re
from torch import optim
from torch import nn
from utils.printing import section_printing_decorator, new_line_appendix_decorator
from feature_extraction.vectorizers import transform_matrix_with_count_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from summarizer import Summarizer, TransformerSummarizer
import spacy
import logging
import sys


def remove_df_rows(df: pd.DataFrame, rows_to_remove_indexes):
    return df[~df.index.isin(df.iloc[rows_to_remove_indexes].index)]


@new_line_appendix_decorator
def print_ngrams_with_the_biggest_count(
    corpus, ngram_length: int, n_top_ngrams: int = 10
):
    top_n_grams = get_top_ngrams(
        corpus, ngram_length=ngram_length, ngrams_limit=n_top_ngrams
    )
    print(f"Top {n_top_ngrams} {ngram_length}-grams:")
    for index, ngram in enumerate(top_n_grams):
        print(f"\t{index + 1}. '{ngram[0]}' [{ngram[1]}]")


def print_top_words_for_all_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join(
            [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        )
        print(message)
    print()


@section_printing_decorator
def show_top_ngrams(corpus):
    print("1. TOP N-GRAMS")
    print()
    for i in range(1, 7):
        print_ngrams_with_the_biggest_count(corpus=corpus, ngram_length=i)


@section_printing_decorator
def show_basic_facts(corpus):
    print("2. BASIC FACTS")
    print()
    facts_extractor = FactsExtractor(corpus=corpus)
    facts_extractor.show_basic_facts()


@section_printing_decorator
def show_most_relevant_sentences(
    normalized_tweets_as_strings: List[str],
    df_before_transformation: pd.DataFrame,
    sentences: List[List[str]],
):
    print("3. MOST RELEVANT TWEETS")
    print()
    tfidf_df = create_df_with_tfidf_feature_vectors(
        corpus=normalized_tweets_as_strings, maximum_number_of_features=100,
    )
    TweetsSummarizer.generate_tweets_summary_based_on_page_rank_and_random_sample(
        tfidf_tweets=tfidf_df,
        df_before_transformation=df_before_transformation,
        sentences=sentences,
    )


def show_summaries_generated_with_transformers(df: pd.DataFrame) -> None:
    print("4. TWEETS SUMMARIES GENERATED WITH TRANSFORMERS")
    print()
    roberta_model_name = "roberta-large"
    gpt2_model_name = "gpt2-large"
    sample_tweets = df[LIV_WAT_TEXT_COLUMN_NAME].sample(n=7500, random_state=0)
    min_sentence_length = 10
    max_sentence_length = 150
    number_of_summary_sentences = 7
    merged_tweets = ""
    for tweet in sample_tweets:
        merged_tweets += f" {tweet}"
        if tweet[-1] not in {"?", ".", "!"}:
            merged_tweets += "."
    print("ffffff", merged_tweets[:400])
    roberta_model = TransformerSummarizer(
        transformer_type="Roberta", transformer_model_key=roberta_model_name,
    )
    gpt_2_model = TransformerSummarizer(
        transformer_type="GPT2", transformer_model_key=gpt2_model_name
    )
    roberta_summary = roberta_model(
        merged_tweets,
        min_length=min_sentence_length,
        max_length=max_sentence_length,
        num_sentences=number_of_summary_sentences,
    )
    print()
    print(roberta_summary)
    print()

    gpt_2_summary = gpt_2_model(
        merged_tweets,
        min_length=min_sentence_length,
        max_length=max_sentence_length,
        num_sentences=number_of_summary_sentences,
    )
    print()
    print(gpt_2_summary)
    print()


@section_printing_decorator
def show_topics_modeled_with_nmf(
    tfidf, original_tweets, aggregated_tfidf, tfidf_vectorizer,
):
    pd.set_option("display.max_colwidth", -1)
    print("5. TOPIC MODELING")
    print()
    max_iter = 1500
    n_of_topics = 4
    alpha = 0.02
    l1_ratio = 0.6
    print(f"NMF: alpha: {alpha}, l1_ratio: {l1_ratio}, max_iter: {max_iter}")
    nmf = NMF(n_components=n_of_topics, alpha=alpha, l1_ratio=l1_ratio,).fit(
        aggregated_tfidf
    )

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words_for_all_topics(nmf, tfidf_feature_names, 15)
    tfidf_topic_similarity = nmf.transform(tfidf)
    minimal_similarity_threshold = 0.015
    tweets_with_topic_assignment = pd.DataFrame(
        np.apply_along_axis(
            lambda row: (max(row), int(np.argmax(row)))
            if max(row) > minimal_similarity_threshold
            else (None, None),
            1,
            tfidf_topic_similarity,
        ),
        columns=["topic_value", "topic"],
    )
    tweets_with_topic_assignment.insert(
        loc=0, column="tweet", value=original_tweets,
    )
    print(
        len(
            tweets_with_topic_assignment[
                (tweets_with_topic_assignment["topic"] == 0)
                | (tweets_with_topic_assignment["topic"] == 1)
                | (tweets_with_topic_assignment["topic"] == 2)
                | (tweets_with_topic_assignment["topic"] == 3)
            ]
        )
    )
    # top tweets per topic
    for topic_index in range(n_of_topics):
        print()
        print(f"=====TOP FOR TOPIC {topic_index}=====")
        topic_tweets = tweets_with_topic_assignment[
            tweets_with_topic_assignment["topic"] == topic_index
        ]
        top_topic_tweets = topic_tweets.sort_values(by="topic_value", ascending=False)[
            :10
        ]
        print(top_topic_tweets)

    print(
        tweets_with_topic_assignment[
            (tweets_with_topic_assignment["topic"] != 0)
            & (tweets_with_topic_assignment["topic"] != 1)
            & (tweets_with_topic_assignment["topic"] != 2)
            & (tweets_with_topic_assignment["topic"] != 3)
        ][:10]
    )


# todo roberta, opengpt3
# todo time frames


def aggregate_tweets(
    tweets_to_aggregate, aggregated_record_tweet_count: int = 5,
):
    aggregated_tweets = []
    counter = 0
    while True:
        start = counter * aggregated_record_tweet_count
        stop = (counter + 1) * aggregated_record_tweet_count
        if stop < len(tweets_to_aggregate):
            aggregated_tweets.append(" ".join(tweets_to_aggregate[start:stop]))
        else:
            aggregated_tweets.append(" ".join(tweets_to_aggregate[start:]))
            break
        counter += 1
    return aggregated_tweets


def run_liverpool_watford_analysis():
    # data loading
    configs = ConfigLoader.load()
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )

    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    transformed_corpus_without_emoticons = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets,
        hyper_parameters_config=configs.settings["analysis_without_emoticons"],
    )
    transformed_corpus_with_emoticons = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets,
        hyper_parameters_config=configs.settings["analysis_with_emoticons"],
    )
    normalized_tweets_as_token_lists = transformed_corpus_without_emoticons.corpus
    df = remove_df_rows(
        df=df,
        rows_to_remove_indexes=transformed_corpus_without_emoticons.indexes_of_removed_tweets,
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    flattened_and_normalized_tweets = []
    normalized_tweets_as_strings = []
    for tweet in normalized_tweets_as_token_lists:
        flattened_and_normalized_tweets.extend(tweet)
        normalized_tweets_as_strings.append(" ".join(tweet))
    tfidf_vectorizer = TfidfVectorizer(
        token_pattern=r"\S+", stop_words="english", min_df=20, max_features=1350,
    )
    tfidf = tfidf_vectorizer.fit_transform(normalized_tweets_as_strings)
    aggregated_tweets = aggregate_tweets(
        tweets_to_aggregate=normalized_tweets_as_strings,
    )
    tfidf_aggregated_tweets = tfidf_vectorizer.fit_transform(aggregated_tweets)

    # top n-grams
    # show_top_ngrams(corpus=normalized_tweets_as_token_lists)

    # facts extraction
    # show_basic_facts(corpus=tweets)

    # summarization
    # show_most_relevant_sentences(
    #     normalized_tweets_as_strings=normalized_tweets_as_strings,
    #     df_before_transformation=df,
    #     sentences=normalized_tweets_as_token_lists,
    # )
    # show_summaries_generated_with_transformers(df)

    # topic modeling
    show_topics_modeled_with_nmf(
        tfidf=tfidf,
        original_tweets=tweets,
        aggregated_tfidf=tfidf_aggregated_tweets,
        tfidf_vectorizer=tfidf_vectorizer,
    )

    # todo improve summarization
    # todo refactor
    # todo emotions, transfer learning


if __name__ == "__main__":
    run_liverpool_watford_analysis()
