import logging
import os.path
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk import SklearnClassifier, MaxentClassifier, classify
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from configs.config_schema import PathsConfig
from data.column_names import LABEL_COLUMN_NAME
from data.utils import DataSet
from learning_datasets.index import TransferLearningDataSet
from learning_datasets.loader import DatasetLoader
from sentiment.labeler import Labeler
from utils.printing import section_printing_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_logger")


def get_model_accuracies(original_labeled_df, model_labeled_df, paths_config: PathsConfig):
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
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"],
    )
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(paths_config.results_dir, "confusion_matrix_vader"))
    plt.show()

    matching_label_rows = original_labeled_df[
        (original_labeled_df["label"] == model_labeled_df["label"])
    ]

    average_accuracy = len(matching_label_rows) / len(original_labeled_df)
    return label_accuracies, average_accuracy, label_counts


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

def _test_with_lstm(
        train_dataset: DataSet,
        test_dataset: DataSet,
        main_test_dataset: DataSet,
):
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
    test_predictions = model.predict(padded_test_tweets)
    transfer_predictions = model.predict(padded_main_tweets)

    print("test", len(test_predictions))
    print("transfer", len(transfer_predictions))
    test_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in np.argmax(test_predictions,axis=1)
        ]
    )
    transfer_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in np.argmax(transfer_predictions,axis=1)
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
    test_predictions = model.predict(padded_test_tweets)
    transfer_predictions = model.predict(padded_main_tweets)

    print("test", len(test_predictions))
    print("transfer", len(transfer_predictions))
    test_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in np.argmax(test_predictions,axis=1)
        ]
    )
    transfer_predictions_with_correct_class_names = pd.Series(
        [
            prediction if prediction != 2 else -1
            for prediction in np.argmax(transfer_predictions,axis=1)
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



def test_different_models_for_dataset(
    tl_dataset: TransferLearningDataSet,
    main_test_dataset: DataSet,
    convert_emojis_to_text: bool
):
    logger.disabled = True

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
    df = pd.DataFrame.from_dict(
        {
            "random_forrest": random_forrest_test_results.accuracies,
            "logistic_regression": logistic_regression_test_results.accuracies,
            "bagging": bagging_test_results.accuracies,
            "maxent": maxent_test_results.accuracies,
            "sgd": sgd_test_results.accuracies,
            "lstm_results": lstm_results.accuracies,
            "cnn_results": cnn_results.accuracies,
        },
        orient="index",
        columns=["test_accuracy", "transfer_accuracy"]
    )
    print(df)
    print()
    return df


def calculate_trained_models_accuracies(
        dataset: DataSet,
        config_file_path: str,
        convert_emojis_to_text = True
):
    dataframes, dataframes_keys = [], []
    learning_data = DatasetLoader.load_learning_datasets(
        config_file_path=config_file_path, dataset=dataset, convert_emojis_to_text=convert_emojis_to_text)
    for index, tl_dataset in enumerate(learning_data):
        df = test_different_models_for_dataset(
            tl_dataset=tl_dataset,
            main_test_dataset=dataset,
            convert_emojis_to_text=convert_emojis_to_text,
        )
        dataframes.append(df)
        dataframe_key = f"DataSet {index + 1}"
        dataframes_keys.append(dataframe_key)

    accuracies_df = pd.concat(dataframes, keys=dataframes_keys,)
    return accuracies_df


@section_printing_decorator("SENTIMENT ANALYSIS")
def show_sentiment_analysis_accuracies_results(
        dataset: DataSet,
        config_file_path: str,
        paths_config: PathsConfig,
):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    start_time = time.time()
    accuracies_df = calculate_trained_models_accuracies(
        dataset=dataset, config_file_path=config_file_path,
    )
    print(accuracies_df)
    # for threshold_value in np.arange(0.05, 0.65, 0.05):
    print((time.time() - start_time) / 60)

    threshold_value = 0.05
    df_senti_word_net = Labeler.get_data_sentiment_with_senti_word_net(
        dataset.initial_df_with_emojis_converted_to_text, threshold_value
    )
    senti_word_net_accuracies = get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_senti_word_net,
        paths_config=paths_config
    )

    df_text_blob = Labeler.get_data_sentiment_with_text_blob(
        dataset.initial_df, threshold_value
    )
    text_blob_accuracies = get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_text_blob,
        paths_config=paths_config
    )

    df_vader = Labeler.get_data_sentiment_with_vader(
        dataset.initial_df, threshold_value
    )
    vader_accuracies = get_model_accuracies(
        original_labeled_df=dataset.initial_df_with_emojis_converted_to_text,
        model_labeled_df=df_vader,
        paths_config=paths_config
    )
    print(f"------------{threshold_value:.2f}, ----------------")
    print(f"SENTI: {senti_word_net_accuracies[0]} => {senti_word_net_accuracies[1]}, {senti_word_net_accuracies[2]}")
    print(f"BLOB: {text_blob_accuracies[0]} => {text_blob_accuracies[1]}, {text_blob_accuracies[2]}")
    print(f"VADER: {vader_accuracies[0]} => {vader_accuracies[1]}, {vader_accuracies[2]}")
    print("----------------------------")
