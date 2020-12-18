from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from summarizer import TransformerSummarizer
from configs.config_loader import ConfigLoader
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover
from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME, CREATED_AT_COLUMN_NAME
from data.loaders import DataLoader
from data.paths import (
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH,
)
import os
import math
from collections import Counter
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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import logging
import time
from pprint import pprint
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf
import datetime

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
    for i in range(1, 5):
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
        background_color="white", mask=mask, #color_func=color_func,
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
    # print("ffffff", merged_tweets[:400])
    # todo test normalized text
    roberta_model = TransformerSummarizer(
        transformer_type="Roberta", transformer_model_key=roberta_model_name,
    )
    # gpt_2_model = TransformerSummarizer(
    #     transformer_type="GPT2", transformer_model_key=gpt2_model_name
    # )
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
def show_topics_modeled_with_nmf(dataset: DataSet) -> pd.DataFrame:
    pd.set_option("display.max_colwidth", -1)
    print("5. TOPIC MODELING")
    print()
    max_iter = 1500
    n_of_topics = 4
    alpha = 0.02
    l1_ratio = 0.6
    print(f"NMF: alpha: {alpha}, l1_ratio: {l1_ratio}, max_iter: {max_iter}")
    nmf = NMF(
        n_components=n_of_topics, alpha=alpha, l1_ratio=l1_ratio,  # max_iter=max_iter
    ).fit(dataset.tfidf_for_aggregated_tweets)
    tfidf_feature_names = dataset.tfidf_vectorizer.get_feature_names()
    print_top_words_for_all_topics(nmf, tfidf_feature_names, 15)
    tfidf_topic_similarity = nmf.transform(dataset.tfidf)
    minimal_similarity_threshold = 0.015  # domyslnie 0.015
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
    tweets_with_topic_assignment.insert(loc=0, column="tweet", value=dataset.initial_tweets_array)
    tweets_with_topic_assignment.insert(
        loc=1, column=CREATED_AT_COLUMN_NAME, value=dataset.initial_df[CREATED_AT_COLUMN_NAME].to_numpy()
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
        top_topic_tweets = topic_tweets.sort_values(by="topic_value", ascending=False)[:15]
        print(top_topic_tweets)

    print(
        tweets_with_topic_assignment[
            (tweets_with_topic_assignment["topic"] != 0)
            & (tweets_with_topic_assignment["topic"] != 1)
            & (tweets_with_topic_assignment["topic"] != 2)
            & (tweets_with_topic_assignment["topic"] != 3)
        ][:10]
    )
    return tweets_with_topic_assignment


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
        convert_emojis_to_text: bool,
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
            token_pattern=r"\S+", stop_words="english", max_features=1000,
        )
    )
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.tfidf.toarray(),#[:,:,None],
        dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME],
        test_size=0.1,
    )
    if not convert_emojis_to_text:
        x_main = dataset.tfidf_vectorizer.fit_transform(main_test_dataset.normalized_tweets_as_strings)
    else:
        x_main = dataset.tfidf_vectorizer.fit_transform(
            main_test_dataset.normalized_tweets_as_demojized_strings
        ).toarray()#[:,:,None]
    return x_train, x_test, x_main, y_train, y_test


from dataclasses import dataclass


@dataclass
class TransferLearningDataSetInfo:
    path: str
    text_column_name: str
    label_column_name: str
    sentiment_label_conversion_map: Optional[Dict] = None


@dataclass
class TransferLearningDataSet:
    train_data_nltk: DataSet
    test_data_nltk: DataSet
    train_data_tfidf: List
    train_data_tfidf_labels: List
    test_data_tfidf: List
    test_data_tfidf_labels: List
    main_test_data_tfidf: List


@dataclass
class ModelTestResults:
    test_accuracy: Optional[float]
    transfer_accuracy: float
    transfer_predictions: pd.Series

    @property
    def accuracies(self) -> Tuple[float, float]:
        return self.test_accuracy, self.transfer_accuracy


def _train_and_test_with_nltk_model(
    classifier,
    train_dataset: DataSet,
    test_dataset: DataSet,
    main_test_dataset: DataSet,
    convert_emojis_to_text: bool,
    **cutoffs
) -> ModelTestResults:
    trained_classifier = classifier.train(
        train_dataset.prepare_for_nltk_classifier(),
        **cutoffs
    )
    class_name = (
        trained_classifier.__class__.__name__
        if not isinstance(classifier, SklearnClassifier)
        else trained_classifier._clf
    )
    logger.info(f"-------{class_name}-------")

    prepared_test_dataset = test_dataset.prepare_for_nltk_classifier(convert_emojis_to_text)
    prepared_main_test_dataset = main_test_dataset.prepare_for_nltk_classifier(
        convert_emojis_to_text
    )

    test_accuracy = classify.accuracy(trained_classifier, prepared_test_dataset)
    transfer_accuracy = classify.accuracy(trained_classifier, prepared_main_test_dataset)

    transfer_predictions = trained_classifier.classify_many(
        featureset for featureset, label in prepared_main_test_dataset
    )
    logger.info(f"Accuracy is: {test_accuracy}")
    logger.info(f"Transfer accuracy is: {transfer_accuracy}")

    # classifier.show_most_informative_features(10)
    return ModelTestResults(
        test_accuracy=test_accuracy,
        transfer_accuracy=transfer_accuracy,
        transfer_predictions=transfer_predictions
    )


def _test_with_voting(
    main_test_dataset: DataSet,
    models_test_results: List[ModelTestResults],
    enable_weighting: bool = False,
    top_n_models: int = 3,
):
    logger.info("--------Voting------")
    labels = main_test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME]

    voted_predictions = []
    for index in range(len(labels)):
        vote_results = {-1: 0, 0: 0, 1: 0}
        top_models_test_results = sorted(
            models_test_results,
            key=lambda model_test_result: model_test_result.transfer_accuracy
        )[-top_n_models:]
        for model_test_results in top_models_test_results:
            prediction = model_test_results.transfer_predictions[index]
            if enable_weighting:
                vote_results[prediction] += model_test_results.transfer_accuracy
            else:
                vote_results[prediction] += 1
        sorted_results = sorted(vote_results.items(), key=lambda item: item[1], reverse=True)
        vote_result = sorted_results[0][0]
        voted_predictions.append(vote_result)
    correct_predictions = [
        voted_prediction == main_test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME].loc[index]
        for index, voted_prediction in enumerate(voted_predictions)
    ]
    transfer_accuracy = sum(correct_predictions) / len(voted_predictions)
    return ModelTestResults(
        test_accuracy=None,
        transfer_accuracy=transfer_accuracy,
        transfer_predictions=pd.Series(voted_predictions),
    )


def _test_with_simple_neural_network(
        tl_dataset,
        main_test_dataset: DataSet,
):
    logger.info("------Custom Neural Network------")
    labels = main_test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME]
    model = Sequential()
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    y_train = to_categorical(tl_dataset.train_data_tfidf_labels, num_classes=3)
    y_test = to_categorical(tl_dataset.test_data_tfidf_labels, num_classes=3)
    labels = to_categorical(labels, num_classes=3)
    model.fit(
        tl_dataset.train_data_tfidf,
        y_train,
        epochs=20,
        use_multiprocessing=True,
        # steps_per_epoch=100,
        callbacks=[EarlyStopping(monitor='accuracy', mode='max', min_delta=1, patience=2)],
    )
    test_accuracy = model.evaluate(
        tl_dataset.test_data_tfidf,
        y_test
    )[1]
    test_predictions = model.predict_classes(tl_dataset.test_data_tfidf)
    transfer_predictions = model.predict_classes(tl_dataset.main_test_data_tfidf)

    transfer_accuracy = model.evaluate(
        tl_dataset.main_test_data_tfidf,
        labels
    )[1]
    print("test", Counter(test_predictions))
    print("transfer", Counter(transfer_predictions))
    transfer_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in transfer_predictions
        ]
    )
    return ModelTestResults(
        test_accuracy=test_accuracy,
        transfer_accuracy=transfer_accuracy,
        transfer_predictions=transfer_predictions_with_correct_class_names,
    )


def _test_with_lstm(
        train_dataset: DataSet,
        test_dataset: DataSet,
        main_test_dataset: DataSet,
):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    logger.info("------LSTM Neural Network------")
    tokenizer = Tokenizer(num_words=1500)
    tokenizer.fit_on_texts(train_dataset.normalized_tweets_as_demojized_strings)
    train_sequences = tokenizer.texts_to_sequences(train_dataset.normalized_tweets_as_demojized_strings)
    test_sequences = tokenizer.texts_to_sequences(test_dataset.normalized_tweets_as_demojized_strings)
    main_sequences = tokenizer.texts_to_sequences(main_test_dataset.normalized_tweets_as_demojized_strings)
    labels = main_test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME]
    padded_tweets = pad_sequences(train_sequences, maxlen=45)
    padded_test_tweets = pad_sequences(test_sequences, maxlen=45)
    padded_main_tweets = pad_sequences(main_sequences, maxlen=45)
    y_train = to_categorical(train_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME], num_classes=3)

    model = Sequential()
    model.add(layers.Embedding(1500, 45))
    model.add(layers.LSTM(9, dropout=0.1))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        padded_tweets,
        y_train,
        epochs=20,
        use_multiprocessing=True,
        # steps_per_epoch=1000,
        callbacks=[EarlyStopping(monitor='accuracy', mode='max', min_delta=1, patience=2)],
    )
    test_predictions = model.predict_classes(padded_test_tweets)
    transfer_predictions = model.predict_classes(padded_main_tweets)

    print("test", Counter(test_predictions))
    print("transfer", Counter(transfer_predictions))
    test_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in test_predictions
        ]
    )
    transfer_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in transfer_predictions
        ]
    )
    correct_test_predictions = [
        test_prediction == list(test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME])[index]
        for index, test_prediction in enumerate(test_predictions_with_correct_class_names)
    ]
    test_accuracy = sum(correct_test_predictions) / len(test_predictions_with_correct_class_names)
    correct_predictions = [
        transfer_prediction == labels[index]
        for index, transfer_prediction in enumerate(transfer_predictions_with_correct_class_names)
    ]
    transfer_accuracy = sum(correct_predictions) / len(transfer_predictions_with_correct_class_names)
    return ModelTestResults(
        test_accuracy=test_accuracy,
        transfer_accuracy=transfer_accuracy,
        transfer_predictions=transfer_predictions_with_correct_class_names,
    )


def _test_with_cnn(
        train_dataset: DataSet,
        test_dataset: DataSet,
        main_test_dataset: DataSet,
):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    logger.info("------CNN-----")
    tokenizer = Tokenizer(num_words=1500)
    tokenizer.fit_on_texts(train_dataset.normalized_tweets_as_demojized_strings)
    train_sequences = tokenizer.texts_to_sequences(train_dataset.normalized_tweets_as_demojized_strings)
    test_sequences = tokenizer.texts_to_sequences(test_dataset.normalized_tweets_as_demojized_strings)
    main_sequences = tokenizer.texts_to_sequences(main_test_dataset.normalized_tweets_as_demojized_strings)
    labels = main_test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME]
    padded_tweets = pad_sequences(train_sequences, maxlen=50)
    padded_test_tweets = pad_sequences(test_sequences, maxlen=50)
    padded_main_tweets = pad_sequences(main_sequences, maxlen=50)
    y_train = to_categorical(train_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME], num_classes=3)

    model = Sequential()
    model.add(layers.Embedding(1500, 50))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(3, activation='softmax'))


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        padded_tweets,
        y_train,
        epochs=20,
        use_multiprocessing=True,
        # steps_per_epoch=1000,
        callbacks=[EarlyStopping(monitor='accuracy', mode='max', min_delta=0.1, patience=2)],
    )
    test_predictions = model.predict_classes(padded_test_tweets)
    transfer_predictions = model.predict_classes(padded_main_tweets)

    print("test", Counter(test_predictions))
    print("transfer", Counter(transfer_predictions))
    test_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in test_predictions
        ]
    )
    transfer_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in transfer_predictions
        ]
    )
    correct_test_predictions = [
        test_prediction == list(test_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME])[index]
        for index, test_prediction in enumerate(test_predictions_with_correct_class_names)
    ]
    test_accuracy = sum(correct_test_predictions) / len(test_predictions_with_correct_class_names)
    correct_predictions = [
        transfer_prediction == labels[index]
        for index, transfer_prediction in enumerate(transfer_predictions_with_correct_class_names)
    ]
    transfer_accuracy = sum(correct_predictions) / len(transfer_predictions_with_correct_class_names)
    return ModelTestResults(
        test_accuracy=test_accuracy,
        transfer_accuracy=transfer_accuracy,
        transfer_predictions=transfer_predictions_with_correct_class_names,
    )


TRANSFER_LEARNING_DATASETS_INFO = [
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/cosmos98_twitter-and-reddit-sentimental-analysis-dataset/Twitter_Data.csv",
        text_column_name="clean_text",
        label_column_name="category",
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/abhi8923shriv_tweetsentimentextraction/train.csv",
        text_column_name="text",
        label_column_name="sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/arbazkhan971_product-sentiment-analysis/Participants_Data/Train.csv",
        text_column_name="Product_Description",
        label_column_name="Sentiment",
        sentiment_label_conversion_map={0: 0, 3: 0,  1: -1, 2: 1},
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/bhuwanesh340_predicting-tweet-sentiments/train.csv",
        text_column_name="original_text",
        label_column_name="sentiment_class",
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/ivankunchev_tweet-sentiment-extraction-ml/train_ml.csv",
        text_column_name="text",
        label_column_name="sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/louise2001_extended-train-for-tweet/extended_train.csv",
        text_column_name="text",
        label_column_name="sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/maxjon_complete-tweet-sentiment-extraction-data/tweet_dataset.csv",
        text_column_name="text",
        label_column_name="new_sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/seshurajup_tweet-sentiment-extraction-old/train.csv",
        text_column_name="text",
        label_column_name="sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    TransferLearningDataSetInfo(
        path="/home/maciej_szczypek/Downloads/potential_datasets/vivekrathi055_sentiment-analysis-on-financial-tweets/tweet_sentiment.csv",
        text_column_name="cleaned_tweets",
        label_column_name="sentiment",
    ),
]


def test_different_models_for_dataset(
    tl_dataset: TransferLearningDataSet,
    main_test_dataset: DataSet,
    convert_emojis_to_text: bool
):
    logger.disabled = True

    naive_bayes_test_results = _train_and_test_with_nltk_model(
        classifier=NaiveBayesClassifier,
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
        convert_emojis_to_text=convert_emojis_to_text,
    )
    random_forrest_test_results = _train_and_test_with_nltk_model(
        classifier=SklearnClassifier(
            estimator=RandomForestClassifier(n_estimators=5)
        ),
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
        convert_emojis_to_text=convert_emojis_to_text,
    )
    logistic_regression_test_results = _train_and_test_with_nltk_model(
        classifier=SklearnClassifier(
            estimator=LogisticRegression()
        ),
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
        convert_emojis_to_text=convert_emojis_to_text,
    )
    bagging_test_results = _train_and_test_with_nltk_model(
        classifier=SklearnClassifier(
            estimator=BaggingClassifier(
                n_jobs=4,
                n_estimators=3,
            ),
        ),
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
        convert_emojis_to_text=convert_emojis_to_text,
    )
    maxent_test_results = _train_and_test_with_nltk_model(
        classifier=MaxentClassifier,
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
        convert_emojis_to_text=convert_emojis_to_text,
        min_lldelta=2,
        max_iter=10,
    )
    sgd_test_results = _train_and_test_with_nltk_model(
        classifier=SklearnClassifier(
            SGDClassifier(
                n_jobs=4,
            )
        ),
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
        convert_emojis_to_text=convert_emojis_to_text,
    )
    simple_neural_network_results = _test_with_simple_neural_network(
        tl_dataset=tl_dataset,
        main_test_dataset=main_test_dataset,
    )
    lstm_results = _test_with_lstm(
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
    )
    cnn_results = _test_with_cnn(
        train_dataset=tl_dataset.train_data_nltk,
        test_dataset=tl_dataset.test_data_nltk,
        main_test_dataset=main_test_dataset,
    )
    voting_results = _test_with_voting(
        main_test_dataset=main_test_dataset,
        models_test_results=[
            naive_bayes_test_results,
            random_forrest_test_results,
            logistic_regression_test_results,
            bagging_test_results,
            maxent_test_results,
            sgd_test_results,
            simple_neural_network_results,
            cnn_results,
            lstm_results,
        ],
    )
    weighted_voting_results = _test_with_voting(
        main_test_dataset=main_test_dataset,
        models_test_results=[
            naive_bayes_test_results,
            random_forrest_test_results,
            logistic_regression_test_results,
            bagging_test_results,
            maxent_test_results,
            sgd_test_results,
            simple_neural_network_results,
            cnn_results,
            lstm_results,
        ],
        enable_weighting=True,
    )
    df = pd.DataFrame.from_dict(
        {
            "naive_bayes": naive_bayes_test_results.accuracies,
            "random_forrest": random_forrest_test_results.accuracies,
            "logistic_regression": logistic_regression_test_results.accuracies,
            "bagging": bagging_test_results.accuracies,
            "maxent": maxent_test_results.accuracies,
            "sgd": sgd_test_results.accuracies,
            "simple_neural_network_results": simple_neural_network_results.accuracies,
            "lstm_results": lstm_results.accuracies,
            "cnn_results": cnn_results.accuracies,
            "voting": voting_results.accuracies,
            "weighted_voting": weighted_voting_results.accuracies,
        },
        orient="index",
        columns=["test_accuracy", "transfer_accuracy"]
    )
    print(df)
    print()
    return df


def _calculate_trained_models_accuracies(
        dataset: DataSet,
        transfer_learning_datasets_info: List[TransferLearningDataSetInfo],
):
    dataframes = []
    dataframes_keys = []
    convert_emojis_to_text = True

    for index, tl_dataset_info in enumerate(transfer_learning_datasets_info):
        print(f"*****{tl_dataset_info.path}*****")
        train_dataset, test_dataset = _load_datasets(
            file_path=tl_dataset_info.path,
            text_column_name=tl_dataset_info.text_column_name,
            label_column_name=tl_dataset_info.label_column_name,
            sentiment_label_conversion_map=tl_dataset_info.sentiment_label_conversion_map,
        )
        x_train, x_test, x_main, y_train, y_test = _load_tfidf_datasets(
            file_path=tl_dataset_info.path,
            text_column_name=tl_dataset_info.text_column_name,
            label_column_name=tl_dataset_info.label_column_name,
            main_test_dataset=dataset,
            sentiment_label_conversion_map=tl_dataset_info.sentiment_label_conversion_map,
            convert_emojis_to_text=convert_emojis_to_text,
        )
        tl_dataset = TransferLearningDataSet(
            train_data_nltk=train_dataset,
            test_data_nltk=test_dataset,
            train_data_tfidf=x_train,
            train_data_tfidf_labels=y_train,
            test_data_tfidf=x_test,
            test_data_tfidf_labels=y_test,
            main_test_data_tfidf=x_main,
        )
        df = test_different_models_for_dataset(
            tl_dataset=tl_dataset,
            main_test_dataset=dataset,
            convert_emojis_to_text=convert_emojis_to_text,
        )
        dataframes.append(df)
        dataframe_key = (
            f"DataSet {index + 1} ({len(train_dataset.df_with_normalized_tweets)} records)"
        )
        dataframes_keys.append(dataframe_key)

    accuracies_df = pd.concat(
        dataframes,
        keys=dataframes_keys,
    )
    return accuracies_df


@section_printing_decorator
def show_sentiment_analysis_accuracies_results(
        dataset: DataSet,
        accuracies_df_path_to_load: Optional[str] = None,
        output_accuracies_df_path: Optional[str] = None,
):
    print("6. SENTIMENT ANALYSIS")
    print()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    start_time = time.time()
    if accuracies_df_path_to_load:
        accuracies_df = pd.read_csv(accuracies_df_path_to_load)
        grouped_accuracies_df = accuracies_df.groupby(["Unnamed: 0", "Unnamed: 1"])
        group_keys = grouped_accuracies_df.groups.keys()
        splitted_dfs = [
            grouped_accuracies_df.get_group(key).iloc[:, -3:]
            for key in group_keys
        ]
        accuracies_df = pd.concat(splitted_dfs, keys=group_keys)
    else:
        accuracies_df = _calculate_trained_models_accuracies(
            dataset=dataset,
            transfer_learning_datasets_info=TRANSFER_LEARNING_DATASETS_INFO,
        )
        if output_accuracies_df_path:
            accuracies_df.to_csv(
                output_accuracies_df_path,
            )
    print(accuracies_df)
    # for threshold_value in np.arange(0.05, 0.65, 0.05):
    print((time.time() - start_time) / 60)

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
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_text_blob,
    )

    df_vader = Labeler.get_data_sentiment_with_vader(
        dataset.initial_df, threshold_value
    )
    vader_accuracies = _get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
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


def _get_time_frames(time_series: pd.Series, time_frame_minutes_length: int) -> pd.DataFrame:
    min_time = min(time_series)
    max_time = max(time_series)
    min_time_series_time = min_time.replace(
        second=0,
        minute=(
            math.floor(min_time.minute / time_frame_minutes_length) * time_frame_minutes_length
        )
    )
    max_time_series_minutes = (
            math.ceil(max_time.minute / time_frame_minutes_length) * time_frame_minutes_length
    )
    max_time_series_time = max_time.replace(
        second=0,
        minute=max_time_series_minutes if max_time_series_minutes != 60 else 0,
        hour=max_time.hour if max_time_series_minutes != 60 else max_time.hour + 1
    )
    current_time_frame_start = min_time_series_time
    time_frames = []
    while (
        current_time_frame_start + datetime.timedelta(minutes=time_frame_minutes_length)
        <= max_time_series_time
    ):
        current_time_frame_end = (
            current_time_frame_start
            + datetime.timedelta(minutes=time_frame_minutes_length)
            - datetime.timedelta(seconds=1)
        )
        time_frames.append((current_time_frame_start, current_time_frame_end))
        current_time_frame_start += datetime.timedelta(minutes=time_frame_minutes_length)
    df_time_frames = pd.DataFrame(
        data=time_frames, columns=["start", "end"]
    )
    df_time_frames["start_hour"] = df_time_frames.start.dt.time
    df_time_frames["end_hour"] = df_time_frames.end.dt.time
    df_time_frames[LABEL_COLUMN_NAME] = df_time_frames["start_hour"].astype(str) + " - " + df_time_frames["end_hour"].astype(str)
    return df_time_frames


def _show_topic_statistics(df_topic: pd.DataFrame, time_frames, total_tweets_per_time_frame):
    df = (
        df_topic
        .groupby(["time_frame", "topic"])["tweet"]
        .count()
        .reset_index(name="count")
    )
    df = df.merge(total_tweets_per_time_frame, on="time_frame")
    df["percentage"] = df["count"] / df["time_frame_count"] * 100
    fig, axes = plt.subplots(nrows=2, ncols=2)
    topic_count_series = {}
    topic_percentage_series = {}
    for topic in df["topic"].unique():
        topic_name = f"topic_{int(topic)}"
        topic_count_series[topic_name] = df[df.topic == topic].set_index("time_frame")["count"]
        topic_percentage_series[topic_name] = df[df.topic == topic].set_index("time_frame")["percentage"]

    df_topic_count_series = pd.DataFrame(topic_count_series)
    df_topic_percentage_series = pd.DataFrame(topic_percentage_series)
    df_topic_count_series.plot(ax=axes[0, 0])
    df_topic_percentage_series.plot(ax=axes[0, 1])
    df_topic_count_series.plot.bar(ax=axes[1, 0], stacked=True)
    df_topic_percentage_series.plot.bar(ax=axes[1, 1], stacked=True)
    fig.set_figheight(14)
    fig.set_figwidth(16)
    axes[0, 0].set_xticks(time_frames.index)
    axes[0, 0].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    axes[0, 1].set_xticks(time_frames.index)
    axes[0, 1].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    axes[1, 0].set_xticks(time_frames.index)
    axes[1, 0].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    axes[1, 1].set_xticks(time_frames.index)
    axes[1, 1].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    plt.show()
    print(df)


def _show_sentiment_statistics(df_sentiment: pd.DataFrame, time_frames, total_tweets_per_time_frame):
    df = (
        df_sentiment
        .groupby(["time_frame", LABEL_COLUMN_NAME])[TEXT_COLUMN_NAME]
        .count()
        .reset_index(name="count")
    )
    df = df.merge(total_tweets_per_time_frame, on="time_frame")
    df["percentage"] = df["count"] / df["time_frame_count"] * 100
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sentiment_count_series = {}
    sentiment_percentage_series = {}
    for sentiment in df[LABEL_COLUMN_NAME].unique():
        sentiment_count_series[sentiment] = df[df[LABEL_COLUMN_NAME] == sentiment].set_index("time_frame")["count"]
        sentiment_percentage_series[sentiment] = df[df[LABEL_COLUMN_NAME] == sentiment].set_index("time_frame")["percentage"]

    df_sentiment_count_series = pd.DataFrame(sentiment_count_series)
    df_sentiment_percentage_series = pd.DataFrame(sentiment_percentage_series)
    df_sentiment_count_series.plot(ax=axes[0, 0], ylabel="count", xlabel="time frame")
    df_sentiment_percentage_series.plot(ax=axes[0, 1], ylabel="percentage [%]", xlabel="time frame")
    df_sentiment_count_series.plot.bar(ax=axes[1, 0], stacked=True, ylabel="count", xlabel="time frame")
    df_sentiment_percentage_series.plot.bar(ax=axes[1, 1], stacked=True, ylabel="percentage [%]", xlabel="time frame")
    fig.set_figheight(16)
    fig.set_figwidth(18)
    axes[0, 0].set_xticks(time_frames.index)
    axes[0, 0].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    axes[0, 1].set_xticks(time_frames.index)
    axes[0, 1].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    axes[1, 0].set_xticks(time_frames.index)
    axes[1, 0].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    axes[1, 1].set_xticks(time_frames.index)
    axes[1, 1].set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    plt.show()
    print(df)


@section_printing_decorator
def show_time_frames_analysis(
    df_sentiment: pd.DataFrame,
    df_topic: pd.DataFrame,
    time_frame_minutes_length: int = 30
):
    print("7. TIME FRAME SENTIMENT AND TOPICS ANALYSIS")
    print()
    sentiment_time_series = df_sentiment[CREATED_AT_COLUMN_NAME].apply(pd.to_datetime)
    time_frames = _get_time_frames(
        time_series=sentiment_time_series,
        time_frame_minutes_length=time_frame_minutes_length,
    )

    # sentiment
    sentiment_time_frame_labels = sentiment_time_series.apply(
        lambda time_record: time_frames[
            (time_record >= time_frames["start"])
            & (time_record <= time_frames["end"])
        ].index[0]
    )
    df_sentiment["time_frame"] = sentiment_time_frame_labels
    labeled_df_vader = Labeler.get_data_sentiment_with_vader(
        df=df_sentiment, polarity_absolute_threshold=0.05,
    )
    df_sentiment[LABEL_COLUMN_NAME] = labeled_df_vader[LABEL_COLUMN_NAME]

    # topic
    topic_time_series = df_topic[CREATED_AT_COLUMN_NAME].apply(pd.to_datetime)
    topic_time_frame_labels = topic_time_series.apply(
        lambda time_record: time_frames[
            (time_record >= time_frames["start"])
            & (time_record <= time_frames["end"])
        ].index[0]
    )
    df_topic["time_frame"] = topic_time_frame_labels
    # df_sentiment.to_csv("df_sentiment")
    # df_topic.to_csv("df_topic")

    total_tweets_per_time_frame = (
        df_topic
        .groupby('time_frame')["tweet"]
        .count()
        .reset_index(name="time_frame_count")
    )
    total_tweets_per_time_frame.time_frame_count.plot()
    plt.show()
    # _show_topic_statistics(
    #     df_topic=df_topic,
    #     total_tweets_per_time_frame=total_tweets_per_time_frame,
    #     time_frames=time_frames,
    # )
    _show_sentiment_statistics(
        df_sentiment=df_sentiment,
        total_tweets_per_time_frame=total_tweets_per_time_frame,
        time_frames=time_frames,
    )
    # print(df_sentiment)
    # print(time_frames)


def run_liverpool_watford_analysis():
    # data loading
    configs = ConfigLoader.load()
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    # labeled_df_for_sentiment_analysis_tests = DataLoader.from_csv(
    #     LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH
    # )
    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    dataset_without_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=configs.settings["analysis_without_emoticons"],
    )
    # dataset_with_emoticons = DataSet.create(
    #     df=df.copy(),
    #     hyper_parameters_config=configs.settings["analysis_with_emoticons"],
    # )
    # labeled_dataset_for_sentiment_analysis_tests = DataSet.create(
    #     df=labeled_df_for_sentiment_analysis_tests,
    #     hyper_parameters_config=configs.settings[
    #         "setting_for_sentiment_analysis"
    #     ],
    #     tfidf_vectorizer=TfidfVectorizer(
    #         token_pattern=r"\S+", stop_words="english",
    #     )
    # )

    # top n-grams
    show_top_ngrams(corpus=dataset_without_emoticons.normalized_tweets_as_token_lists)
    show_word_cloud(text=dataset_without_emoticons.flat_text)

    # facts extraction
    show_basic_facts(corpus=dataset_without_emoticons.initial_tweets_array)

    # summarization
    # show_most_relevant_sentences(
    #     normalized_tweets_as_strings=dataset_without_emoticons.normalized_tweets_as_strings,
    #     df_before_transformation=dataset_without_emoticons.initial_df,
    #     sentences=dataset_without_emoticons.normalized_tweets_as_token_lists,
    # )
    # show_summaries_generated_with_transformers(dataset_without_emoticons.initial_df)

    # topic modeling
    # df_with_topic_labels = show_topics_modeled_with_nmf(
    #     dataset=dataset_without_emoticons,
    # )

    # summary
    # show_summaries_generated_with_transformers(dataset_without_emoticons.initial_df)

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # sentiment analysis
    # show_sentiment_analysis_accuracies_results(
    #     dataset=labeled_dataset_for_sentiment_analysis_tests,
    #     accuracies_df_path_to_load="/home/maciej_szczypek/PJATK/master_thesis/python_project/data/auxiliary_files/accuracies_df.csv",
    #     # output_accuracies_df_path="/home/maciej_szczypek/PJATK/master_thesis/python_project/data/auxiliary_files/accuracies_df.csv",
    # )

    # time frame analysis
    # show_time_frames_analysis(
    #     # df_sentiment=dataset_with_emoticons.initial_df_with_emojis_converted_to_text,
    #     # df_topic=df_with_topic_labels,
    #     df_sentiment=pd.read_csv("df_sentiment"),
    #     df_topic=pd.read_csv("df_topic"),
    # )


if __name__ == "__main__":
    run_liverpool_watford_analysis()
