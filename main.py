from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from summarizer import TransformerSummarizer
from configs.config_loader import ConfigLoader
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover
from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME
from data.loaders import DataLoader
from data.paths import (
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH,
)

from sklearn.model_selection import train_test_split
from data.utils import DataSet
from feature_extraction.tfidf import create_df_with_tfidf_feature_vectors
from information_extraction.facts_extractor import FactsExtractor
from information_extraction.keyphrase_extraction import get_top_ngrams
from information_extraction.text_summarization import TweetsSummarizer
from sentiment.labeler import Labeler
from utils.printing import section_printing_decorator, new_line_appendix_decorator
import sklearn
import seaborn as sn
import matplotlib.pyplot as plt
from nltk import classify
from nltk import NaiveBayesClassifier, SklearnClassifier, MaxentClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_logger")


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


def show_word_cloud(text):
    from wordcloud import WordCloud
    from PIL import Image
    import random

    # Create and generate a word cloud image:
    football_mask = np.array(Image.open("data/images/football_mask.png"))
    mask = np.array(Image.open("data/images/footballer_mask.jpg"))

    def color_func(*args, **kwargs):
        return f"rgb({random.randint(0, 230)}, {random.randint(0, 230)}, {random.randint(0, 230)})"

    wordcloud = WordCloud(
        background_color="white", mask=mask, color_func=color_func,
    ).generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    wordcloud.to_file("data/images/wordcloud.png")


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
    sample_tweets = df[TEXT_COLUMN_NAME].sample(n=10000, random_state=0)
    min_sentence_length = 10
    max_sentence_length = 100
    number_of_summary_sentences = 15
    merged_tweets = ""
    for tweet in sample_tweets:
        merged_tweets += f" {tweet}"
        if tweet[-1] not in {"?", ".", "!"}:
            merged_tweets += "."
    print("ffffff", merged_tweets[:400])
    # todo test normalized text
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

    # gpt_2_summary = gpt_2_model(
    #     merged_tweets,
    #     min_length=min_sentence_length,
    #     max_length=max_sentence_length,
    #     num_sentences=number_of_summary_sentences,
    # )
    # print()
    # print(gpt_2_summary)
    # print()


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


def _get_model_accuracies(original_labeled_df, model_labeled_df):
    label_accuracies = []
    labels = [-1, 0, 1]
    label_counts = []
    for label in labels:
        number_of_label_matching_records = len(
            original_labeled_df[original_labeled_df["label"] == label]
        )
        matching_label_rows = original_labeled_df[
            (original_labeled_df["label"] == model_labeled_df["label"])
            & (model_labeled_df["label"] == label)
        ]
        label_counts.append(len(
            model_labeled_df[model_labeled_df["label"] == label]
        ))
        accuracy = len(matching_label_rows) / number_of_label_matching_records
        label_accuracies.append(round(accuracy, 2))
    confusion_matrix = sklearn.metrics.confusion_matrix(
        original_labeled_df["label"], model_labeled_df["label"], labels=labels,
    )
    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    sn.set(font_scale=1.4)
    cmap = sn.color_palette("rocket_r", as_cmap=True)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=cmap, fmt="d")
    plt.show()

    matching_label_rows = original_labeled_df[
        (original_labeled_df["label"] == model_labeled_df["label"])
    ]

    average_accuracy = len(matching_label_rows) / len(original_labeled_df)
    return label_accuracies, average_accuracy, label_counts


def _load_datasets(
        file_path: str,
        text_column_name: str,
        label_column_name: str,
        sentiment_label_conversion_map: Optional[Dict[Any, int]]
) -> Tuple[DataSet, DataSet]:
    configs = ConfigLoader.load()
    df = pd.read_csv(file_path)
    df = df.rename(
        columns={
            text_column_name: TEXT_COLUMN_NAME,
            label_column_name: LABEL_COLUMN_NAME,
        }
    )
    if sentiment_label_conversion_map:
        df[LABEL_COLUMN_NAME] = df[LABEL_COLUMN_NAME].replace(
            sentiment_label_conversion_map
        )
    df = df.dropna()
    train_df, test_df = train_test_split(df, test_size=0.1)
    train_dataset = DataSet.create(
        df=train_df,
        hyper_parameters_config=configs.settings["setting_for_sentiment_analysis"]
    )
    test_dataset = DataSet.create(
        df=test_df,
        hyper_parameters_config=configs.settings["setting_for_sentiment_analysis"]
    )
    return train_dataset, test_dataset


def _load_tfidf_datasets(
        file_path: str,
        text_column_name: str,
        label_column_name: str,
        main_test_dataset: DataSet,
        sentiment_label_conversion_map: Optional[Dict[Any, int]]
):
    configs = ConfigLoader.load()
    df = pd.read_csv(file_path)
    df = df.rename(
        columns={
            text_column_name: TEXT_COLUMN_NAME,
            label_column_name: LABEL_COLUMN_NAME,
        }
    )
    if sentiment_label_conversion_map:
        df[LABEL_COLUMN_NAME] = df[LABEL_COLUMN_NAME].replace(
            sentiment_label_conversion_map
        )
    df = df.dropna()
    dataset = DataSet.create(
        df=df,
        hyper_parameters_config=configs.settings["setting_for_sentiment_analysis"],
        tfidf_vectorizer=TfidfVectorizer(
            token_pattern=r"\S+", stop_words="english", max_features=1500,
        )
    )
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.tfidf,
        dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME],
        test_size=0.1,
    )
    # X_main = dataset.tfidf_vectorizer.fit_transform(main_test_dataset.normalized_tweets_as_strings)
    X_main = dataset.tfidf_vectorizer.fit_transform(main_test_dataset.normalized_tweets_as_demojized_strings)
    return X_train, X_test, X_main, y_train, y_test


from dataclasses import dataclass


@dataclass
class TransferLearningDataSet:
    path: str
    text_column_name: str
    label_column_name: str
    sentiment_label_conversion_map: Optional[Dict] = None


def _build_and_train_with_nltk_model(
        classifier,
        train_dataset: DataSet,
        test_dataset: DataSet,
        main_test_dataset: DataSet,
        **cutoffs
) -> Tuple[float, float]:
    trained_classifier = classifier.train(
        train_dataset.prepare_for_nltk_classifier(),
        **cutoffs
    )
    test_accuracy = classify.accuracy(trained_classifier, test_dataset.prepare_for_nltk_classifier())
    transfer_accuracy = classify.accuracy(trained_classifier, main_test_dataset.prepare_for_nltk_classifier())
    logger.info(f"Accuracy is: {test_accuracy}")
    logger.info(f"Transfer accuracy is: {transfer_accuracy}")
    # classifier.show_most_informative_features(10)
    return test_accuracy, transfer_accuracy


def _test_with_naive_bayes(
        train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
) -> Tuple[float, float]:
    logger.info("---------NAIVE BAYES-------")
    classifier = NaiveBayesClassifier
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier=classifier,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


def _test_with_maxent(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    logger.info("--------Maxent------")
    classifier = MaxentClassifier
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset, max_iter=10
    )
    return test_accuracy, transfer_accuracy


def _test_with_random_forest(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    logger.info("---------Random Forest------")
    sklearn_classifier = RandomForestClassifier()
    classifier = SklearnClassifier(estimator=sklearn_classifier)
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


def _test_with_logistic_regression(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    logger.info("--------Logistic Regression------")
    sklearn_classifier = LogisticRegression()
    classifier = SklearnClassifier(estimator=sklearn_classifier)
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


def _test_with_logistic_regression_cv(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    logger.info("--------Logistic Regression CV------")
    sklearn_classifier = LogisticRegressionCV()
    classifier = SklearnClassifier(estimator=sklearn_classifier)
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


def _test_with_bagging(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    logger.info("--------Bagging------")
    sklearn_classifier = BaggingClassifier()
    classifier = SklearnClassifier(estimator=sklearn_classifier)
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


def _test_with_voting(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    # todo implement custom version
    logger.info("--------Voting------")
    clf1 = LogisticRegression(multi_class='multinomial')
    clf2 = RandomForestClassifier()
    clf3 = SGDClassifier()
    sklearn_classifier = VotingClassifier([('lr', clf1), ('rf', clf2), ('sgd', clf3)], voting='hard')
    classifier = SklearnClassifier(estimator=sklearn_classifier)
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


def _test_with_sgd(
    train_dataset: DataSet, test_dataset: DataSet, main_test_dataset: DataSet,
):
    logger.info("--------SGD------")
    sklearn_classifier = SGDClassifier()
    classifier = SklearnClassifier(estimator=sklearn_classifier)
    test_accuracy, transfer_accuracy = _build_and_train_with_nltk_model(
        classifier, train_dataset, test_dataset, main_test_dataset,
    )
    return test_accuracy, transfer_accuracy


TRANSFER_LEARNING_DATASETS = [
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/cosmos98_twitter-and-reddit-sentimental-analysis-dataset/Twitter_Data.csv",
    #     text_column_name="clean_text",
    #     label_column_name="category",
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/cosmos98_twitter-and-reddit-sentimental-analysis-dataset/Reddit_Data.csv",
    #     text_column_name="clean_comment",
    #     label_column_name="category",
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/abhi8923shriv_tweetsentimentextraction/train.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/arbazkhan971_product-sentiment-analysis/Participants_Data/Train.csv",
    #     text_column_name="Product_Description",
    #     label_column_name="Sentiment",
    #     sentiment_label_conversion_map={0: 0, 3: 0,  1: -1, 2: 1},
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/bhuwanesh340_predicting-tweet-sentiments/train.csv",
    #     text_column_name="original_text",
    #     label_column_name="sentiment_class",
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/bhuwanesh340_predicting-tweet-sentiments/train.csv",
    #     text_column_name="original_text",
    #     label_column_name="sentiment_class",
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/ivankunchev_tweet-sentiment-extraction-ml/train_ml.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/louise2001_extended-train-for-tweet/extended_train.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    TransferLearningDataSet(
        path="/home/maciej_szczypek/Downloads/potential_datasets/maxjon_complete-tweet-sentiment-extraction-data/tweet_dataset.csv",
        text_column_name="text",
        label_column_name="new_sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/seshurajup_tweet-sentiment-extraction-old/train.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/vivekrathi055_sentiment-analysis-on-financial-tweets/tweet_sentiment.csv",
    #     text_column_name="cleaned_tweets",
    #     label_column_name="sentiment",
    # ),
    # TransferLearningDataSet(
    #     path="/home/maciej_szczypek/Downloads/potential_datasets/karthikcs1_tweet-data-for-sentiment-analysis/etd_full.csv",
    #     text_column_name="TEXT",
    #     label_column_name="VALUE",
    # ),sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
]


def test_different_models_for_dataset(
        tl_dataset: TransferLearningDataSet, main_test_dataset
):
    print(f"*****{tl_dataset.path}*****")
    train_dataset, test_dataset = _load_datasets(
        file_path=tl_dataset.path,
        text_column_name=tl_dataset.text_column_name,
        label_column_name=tl_dataset.label_column_name,
        sentiment_label_conversion_map=tl_dataset.sentiment_label_conversion_map,
    )
    logger.disabled = True

    naive_bayes_accuracies = _test_with_naive_bayes(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    random_forrest_accuracies = _test_with_random_forest(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    logistic_regression_accuracies = _test_with_logistic_regression(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    logistic_regression_cv_accuracies = _test_with_logistic_regression_cv(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    maxent_accuracies = _test_with_maxent(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    sgd_accuracies = _test_with_sgd(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    bagging_accuracies = _test_with_bagging(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_test_dataset=main_test_dataset,
    )
    # voting_accuracies = _test_with_voting(
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     main_test_dataset=main_test_dataset,
    # )
    df = pd.DataFrame.from_dict(
        {
            "naive_bayes": naive_bayes_accuracies,
            "random_forrest": random_forrest_accuracies,
            "logistic_regression": logistic_regression_accuracies,
            "logistic_regression_cv": logistic_regression_cv_accuracies,
            "bagging": bagging_accuracies,
            # "voting": voting_accuracies,
            "maxent": maxent_accuracies,
            "sgd": sgd_accuracies,
        },
        orient="index",
        columns=["test_accuracy", "transfer_accuracy"]
    )
    print(df)
    print()


@section_printing_decorator
def show_sentiment_analysis_results(dataset: DataSet):
    print("6. SENTIMENT ANALYSIS")
    print()
    # todo test aggregated_datasets
    for tl_dataset in TRANSFER_LEARNING_DATASETS:
        test_different_models_for_dataset(tl_dataset=tl_dataset, main_test_dataset=dataset)
    # for threshold_value in np.arange(0.05, 0.65, 0.05):
    return
    threshold_value = 0.05
    df_sentiwordnet = Labeler.get_data_sentiment_with_sentiwordnet(
        dataset.initial_df_with_emojis_converted_to_text, threshold_value
    )
    sentiwordnet_accuracies = _get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_sentiwordnet,
    )

    df_text_blob = Labeler.get_data_sentiment_with_text_blob(
        dataset.initial_df, threshold_value
    )
    text_blob_accuracies = _get_model_accuracies(
        original_labeled_df=dataset.initial_df,
        model_labeled_df=df_text_blob,
    )

    df_vader = Labeler.get_data_sentiment_with_vader(
        dataset.initial_df, threshold_value
    )
    vader_accuracies = _get_model_accuracies(
        original_labeled_df=dataset.initial_df,
        model_labeled_df=df_vader,
    )
    print(f"------------{threshold_value:.2f}, ----------------")
    print(
        f"SENTI: {sentiwordnet_accuracies[0]} => {sentiwordnet_accuracies[1]}, {sentiwordnet_accuracies[2]}"
    )
    print(
        f"BLOB: {text_blob_accuracies[0]} => {text_blob_accuracies[1]}, {text_blob_accuracies[2]}"
    )

    print(
        f"VADER: {vader_accuracies[0]} => {vader_accuracies[1]}, {vader_accuracies[2]}"
    )
    print("----------------------------")

    # create and train model
    # get the best results for original data
    # use transfer learning and compare accuracy to other dictionary based algorithms
    # time frames with sentiment
    # train model on


def run_liverpool_watford_analysis():
    # data loading
    configs = ConfigLoader.load()
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    labeled_df_for_sentiment_analysis_tests = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH
    )
    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    dataset_without_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=configs.settings["analysis_without_emoticons"],
    )
    dataset_with_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=configs.settings["analysis_with_emoticons"],
    )
    labeled_dataset_for_sentiment_analysis_tests = DataSet.create(
        df=labeled_df_for_sentiment_analysis_tests,
        hyper_parameters_config=configs.settings[
            "setting_for_sentiment_analysis"
        ],
        tfidf_vectorizer=TfidfVectorizer(
            token_pattern=r"\S+", stop_words="english",
        )
    )

    # top n-grams
    # show_top_ngrams(corpus=dataset_without_emoticons.normalized_tweets_as_token_lists)
    # show_word_cloud(text=dataset_without_emoticons.flat_text)

    # facts extraction
    # show_basic_facts(corpus=dataset_without_emoticons.initial_tweets_array)

    # summarization
    # show_most_relevant_sentences(
    #     normalized_tweets_as_strings=dataset_without_emoticons.normalized_tweets_as_strings,
    #     df_before_transformation=dataset_without_emoticons.initial_df,
    #     sentences=dataset_without_emoticons.normalized_tweets_as_token_lists,
    # )
    # show_summaries_generated_with_transformers(dataset_without_emoticons.initial_df)

    # topic modeling
    # show_topics_modeled_with_nmf(
    #     tfidf=dataset_without_emoticons.tfidf,
    #     original_tweets=dataset_without_emoticons.initial_tweets_array,
    #     aggregated_tfidf=dataset_without_emoticons.tfidf_for_aggregated_tweets,
    #     tfidf_vectorizer=dataset_without_emoticons.tfidf_vectorizer,
    # )

    # sentiment analysis
    show_sentiment_analysis_results(
        dataset=labeled_dataset_for_sentiment_analysis_tests,
    )
    # todo emotions, transfer learning, time frames
    # todo word cloud
    # todo improve summarization?


if __name__ == "__main__":
    run_liverpool_watford_analysis()
