from typing import List, Dict, Tuple, Any, Optional
import logging
import time
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.utils import to_categorical
from nltk import NaiveBayesClassifier, SklearnClassifier, MaxentClassifier
from nltk import classify
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

from configs.config_loader import ConfigLoader
from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME
from data.utils import DataSet
from sentiment.labeler import Labeler
from utils.printing import section_printing_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_logger")


def get_model_accuracies(original_labeled_df, model_labeled_df):
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
    confusion_matrix_percentage = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    df_cm_percentage = pd.DataFrame(confusion_matrix_percentage, index=labels, columns=labels)
    sn.set(font_scale=1.4)
    cmap = sn.color_palette("rocket_r", as_cmap=True)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=cmap, fmt="d")
    sn.heatmap(
        df_cm_percentage,
        annot=True,
        annot_kws={"size": 16},
        cmap=cmap,
        fmt=".2%",
        xticklabels=["negatywne", "neutralne", "pozytywne"],
        yticklabels=["negatywne", "neutralne", "pozytywne"],
    )
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_vader")
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
        sentiment_label_conversion_map: Optional[Dict[Any, int]],
        config_file_path: str,
) -> Tuple[DataSet, DataSet]:
    configs = ConfigLoader.load(config_file_path)
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
        sentiment_label_conversion_map: Optional[Dict[Any, int]],
        config_file_path: str,
):
    configs = ConfigLoader.load(config_file_path)
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


def calculate_trained_models_accuracies(
        dataset: DataSet,
        transfer_learning_datasets_info: List[TransferLearningDataSetInfo],
        config_file_path: str,
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
            config_file_path=config_file_path,
        )
        x_train, x_test, x_main, y_train, y_test = _load_tfidf_datasets(
            file_path=tl_dataset_info.path,
            text_column_name=tl_dataset_info.text_column_name,
            label_column_name=tl_dataset_info.label_column_name,
            main_test_dataset=dataset,
            sentiment_label_conversion_map=tl_dataset_info.sentiment_label_conversion_map,
            convert_emojis_to_text=convert_emojis_to_text,
            config_file_path=config_file_path,
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
        config_file_path: str,
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
        accuracies_df = calculate_trained_models_accuracies(
            dataset=dataset,
            transfer_learning_datasets_info=TRANSFER_LEARNING_DATASETS_INFO,
            config_file_path=config_file_path,
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
    sentiwordnet_accuracies = get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_sentiwordnet,
    )

    df_text_blob = Labeler.get_data_sentiment_with_text_blob(
        dataset.initial_df, threshold_value
    )
    text_blob_accuracies = get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_text_blob,
    )

    df_vader = Labeler.get_data_sentiment_with_vader(
        dataset.initial_df, threshold_value
    )
    vader_accuracies = get_model_accuracies(
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
