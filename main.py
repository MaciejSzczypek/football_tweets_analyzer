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
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint
from sklearn.preprocessing import normalize
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
from utils.printing import section_printing_decorator, new_line_appendix_decorator
from feature_extraction.vectorizers import transform_matrix_with_count_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD


def remove_df_rows(df: pd.DataFrame, rows_to_remove_indexes):
    return df[~df.index.isin(df.iloc[rows_to_remove_indexes].index)]


@new_line_appendix_decorator
def print_ngrams_with_the_biggest_count(corpus, ngram_length: int, n_top_ngrams: int = 10):
    top_n_grams = get_top_ngrams(corpus, ngram_length=ngram_length, ngrams_limit=n_top_ngrams)
    print(f"Top {n_top_ngrams} {ngram_length}-grams:")
    for index, ngram in enumerate(top_n_grams):
        print(f"\t{index + 1}. '{ngram[0]}' [{ngram[1]}]")


def print_top_words_for_all_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join(
            [
                feature_names[i]
                for i in topic.argsort()[:-n_top_words - 1:-1]
            ]
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
    print("3. EXTRACTIVE SUMMARIZATION - MOST RELEVANT SENTENCES")
    print()
    tfidf_df = create_df_with_tfidf_feature_vectors(
        corpus=normalized_tweets_as_strings,
        maximum_number_of_features=100,
    )
    TweetsSummarizer.generate_tweets_summary_based_on_page_rank_and_random_sample(
        tfidf_tweets=tfidf_df,
        df_before_transformation=df_before_transformation,
        sentences=sentences,
    )


@section_printing_decorator
def show_topics_modeled_with_nmf(
        tfidf,
        original_tweets,
        aggregated_tfidf,
        tfidf_vectorizer,
):
    print("4. TOPIC MODELING")
    print()
    max_iter = 1500
    n_of_topics = 4
    alpha = 0.02
    l1_ratio = 0.6
    print(f"NMF: alpha: {alpha}, l1_ratio: {l1_ratio}, max_iter: {max_iter}")
    nmf = NMF(
        n_components=n_of_topics,
        alpha=alpha,
        l1_ratio=l1_ratio,
    ).fit(aggregated_tfidf)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words_for_all_topics(nmf, tfidf_feature_names, 15)
    tfidf_topic_similarity = nmf.transform(tfidf)
    tweets_with_topic_assignment = pd.DataFrame(
        np.apply_along_axis(
            lambda row: (max(row), int(np.argmax(row))),
            1,
            tfidf_topic_similarity,
        ),
        columns=["topic_value", "topic"]
    )
    tweets_with_topic_assignment.insert(
        loc=0,
        column="tweet",
        value=original_tweets,
    )

    for topic_index in range(n_of_topics):
        print("!!!!", topic_index)
        topic_tweets = tweets_with_topic_assignment[
            tweets_with_topic_assignment["topic"] == topic_index
            ]
        top_topic_tweets = topic_tweets.sort_values(by="topic_value", ascending=False)[:5]
        print(top_topic_tweets)
        print(len(topic_tweets))
    print(
        tweets_with_topic_assignment[tweets_with_topic_assignment["topic_value"] > 0.07].sample(n=50)
    )


# todo roberta, opengpt3
# todo filtrowanie po tematach, pokazac tweety najlepiej pasujace do tematow
# todo time frames


def aggregate_tweets(
        tweets_to_aggregate,
        aggregated_record_tweet_count: int = 5,
):
    aggregated_tweets = []
    counter = 0
    while True:
        start = counter * aggregated_record_tweet_count
        stop = (counter + 1) * aggregated_record_tweet_count
        if stop < len(tweets_to_aggregate):
            aggregated_tweets.append(
                " ".join(tweets_to_aggregate[start:stop])
            )
        else:
            aggregated_tweets.append(
                " ".join(tweets_to_aggregate[start:])
            )
            break
        counter += 1
    return aggregated_tweets


def run_liverpool_watford_analysis():
    # data loading
    configs = ConfigLoader.load()
    initial_df_with_emoticons = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_PATH
    )
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )

    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    corpus_transformation_result = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets, hyper_parameters_config=configs.settings["analysis"],
    )
    normalized_tweets_as_token_lists = corpus_transformation_result.corpus
    df = remove_df_rows(
        df=df, rows_to_remove_indexes=corpus_transformation_result.indexes_of_removed_tweets
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    flattened_and_normalized_tweets = []
    normalized_tweets_as_strings = []
    for tweet in normalized_tweets_as_token_lists:
        flattened_and_normalized_tweets.extend(tweet)
        normalized_tweets_as_strings.append(" ".join(tweet))
    tfidf_vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",
        stop_words='english',
        min_df=24,
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

    # extractive summarization
    # show_most_relevant_sentences(
    #     normalized_tweets_as_strings=normalized_tweets_as_strings,
    #     df_before_transformation=df,
    #     sentences=normalized_tweets_as_token_lists,
    # )

    # topic modeling
    show_topics_modeled_with_nmf(
        tfidf=tfidf,
        original_tweets=tweets,
        aggregated_tfidf=tfidf_aggregated_tweets,
        tfidf_vectorizer=tfidf_vectorizer,
    )


def alternatives():
    return
    summary = TweetsSummarizer.generate_tweets_summary(
        tweets=normalized_tweets_as_token_lists,
        top_n_grams=top_trigrams,
        threshold=20,
    )
    print(summary)
    flattened_more = ". ".join(normalized_tweets_as_strings[:5000])
    print(summarize(flattened_more, ratio=1, word_count=50))

    # ???
    words_occurences = count_word_occurences(flattened_and_normalized_tweets)
    tweets_scores = score_tweets(tweets=normalized_tweets_as_token_lists,
                                 words_occurence=words_occurences)
    # print(tweets_scores)

    # section topic modeling
    lda_dictionary = corpora.Dictionary(normalized_tweets_as_token_lists)
    print(lda_dictionary.items())
    lda_prepared_corpus = [lda_dictionary.doc2bow(text) for text in
                           normalized_tweets_as_token_lists]
    print("before lda model")

    def show_top_words(model):
        for topic in model.print_topics(num_topics=5, num_words=15):
            words = re.findall("\".*?\"", topic[1])
            print([word.replace('"', '') for word in words])
            # print(topic[1])

    data_samples = normalized_tweets_as_strings
    aggregated_data_samples = []
    aggregated_data_size = 5
    counter = 0
    while True:
        start = counter * aggregated_data_size
        stop = (counter + 1) * aggregated_data_size
        if stop < len(normalized_tweets_as_strings):
            aggregated_data_samples.append(
                " ".join(normalized_tweets_as_strings[start:stop])
            )
        else:
            aggregated_data_samples.append(
                " ".join(normalized_tweets_as_strings[start:])
            )
            break
        counter += 1

    # NMF: alpha: 0.014490563814621549, l1_ratio: 0.6340456783822659, max_iter: 3826, aggregated_data_size : 5
    # 0: liverpool watford lost game 3-0 today beat day time 3 v lose goal player lovren
    # 1: unbeaten run end liverpool's watford ended record season streak liverpool 44 arsenal 44game 49 sarr
    # 2: league win premier champion game liverpool winning season title going team gonna lose point year
    # 3: fan liverpool arsenal team man like season united u invincible invincibles best club know losing

    # NMF: alpha: 0.46808551872103143, l1_ratio: 0.463080086560159, max_iter: 1548
    # Topic  # 0: liverpool fan watford team arsenal lost season game like win today u man lose day
    # Topic  # 1: run unbeaten end liverpool's watford ended 44game 44 3-0 streak record liverpool premier come thrashing
    # Topic  # 2: league premier win champion winning title season unbeaten going game gonna liverpool liverpool's europa team

    tfidf_vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",
        stop_words='english',
        min_df=25,
    )
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    tfidf_agg = tfidf_vectorizer.fit_transform(aggregated_data_samples)
    tf_vectorizer = CountVectorizer(
        token_pattern=r"\S+",
        stop_words='english',
        min_df=10,
    )
    # tf = tf_vectorizer.fit_transform(data_samples)
    tf_agg = tf_vectorizer.fit_transform(aggregated_data_samples)
    for i in range(10):  # tu najlepsze 3 topici
        max_iter = random.randint(1, 5000)
        n_of_topics = random.randint(2, 4)
        alpha = random.random()
        l1_ratio = random.random()
        print(f"NMF: alpha: {alpha}, l1_ratio: {l1_ratio}, max_iter: {max_iter}")
        nmf = NMF(
            n_components=n_of_topics,
            alpha=alpha,
            l1_ratio=l1_ratio,
        ).fit(tfidf_agg)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        print_top_words_for_all_topics(nmf, tfidf_feature_names, 15)
    # for i in range(10):
    #     n_of_topics = random.randint(2, 3)
    #     max_iter = random.randint(1, 5000)
    #     learning_offset = random.randint(1, 1000)
    #     evaluate_every = random.randint(1, 200)
    #     print(f"LDA: learning_offset: {learning_offset}, evaluate_every: {evaluate_every}, max_iter: {max_iter}")
    #     lda = LatentDirichletAllocation(
    #         n_components=n_of_topics,
    #         max_iter=5,
    #         learning_method='online',
    #         learning_offset=50.,
    #     )
    #     lda.fit(tf_agg)
    #     tf_feature_names = tf_vectorizer.get_feature_names()
    #     print_top_words(lda, tf_feature_names, 15)
    # for i in range(10):
    #     n_iter = random.randint(1, 15000)
    #     tol = random.random()
    #     print(f"LSI: n_iter: {n_iter}, tol: {tol}")
    #     lsi = TruncatedSVD(
    #         n_components=2, algorithm='randomized', n_iter=n_iter, tol=tol)
    #     lsi.fit(tf)
    #     tf_feature_names = tf_vectorizer.get_feature_names()
    #     print_top_words(lsi, tf_feature_names, 15)

    # todo Pachinko allocation, aggregated tweets and LDA
    print("PA model")
    pa_model = tp.HPAModel(
        min_df=20,
        k1=1,
        k2=2,
    )
    for sentence in normalized_tweets_as_token_lists:
        pa_model.add_doc(sentence)
    for i in range(0, 250, 25):
        pa_model.train(100)
    print('Log-likelihood: {}'.format(pa_model.ll_per_word))

    print(pa_model.get_sub_topic_dist(0))
    for k in range(pa_model.k2):
        print('Top 10 words of subtopic #{}'.format(k))
        print(([word for word, value in pa_model.get_topic_words(k, top_n=15)]))
    return

    # lsi = LsiModel(
    #     corpus=lda_prepared_corpus,
    #     num_topics=2,
    #     id2word=lda_dictionary,
    #     chunksize=10000,
    #     power_iters=75,
    #     onepass=False,
    #     # extra_samples=1000,
    #     # workers=6,
    # )
    # show_top_words(lsi)
    # return
    def show_top_words_with_randomized_parameters():
        # random_state = random.randint(1, 100)
        passes = random.randint(50, 80)
        iterations = random.randint(10, 40)
        chunksize = 10000  # random.randint(3200, 3800)
        update_every = 5  # random.randint(10, 100)
        print("===========")
        print(
            f"passes: {passes}, iterations: {iterations}, chunksize: {chunksize}, update_every: {update_every}, random_state: ")
        lda_model = LdaMulticore(
            corpus=lda_prepared_corpus,
            num_topics=2,
            id2word=lda_dictionary,
            passes=passes,
            iterations=iterations,
            chunksize=chunksize,
            # random_state=random_state,
            # random_state=1,
            # update_every=update_every
            # workers=6,
        )
        show_top_words(lda_model)

    for i in range(10):
        show_top_words_with_randomized_parameters()

    # priority todo's
    # todo refactor
    # todo emotions, transfer learning

    # less relevant for now:
    # todo improve most relevant sentences
    # todo improve topic modeling
    # todo improve summarization


if __name__ == "__main__":
    run_liverpool_watford_analysis()
