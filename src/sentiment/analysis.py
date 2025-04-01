import enum
import logging
from dataclasses import dataclass
from typing import List, Union, Callable

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
from transformers import pipeline

from data.column_names import LABEL_COLUMN_NAME
from data.utils import DataSet
from learning_datasets.loader import DatasetLoader
from sentiment.labeler import Labeler
from utils.printing import section_printing_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_logger")


class Model(enum.Enum):
    LLM = "LLM"
    LSTM = "LSTM"
    VADER = "VADER"


@dataclass
class ModelInfo:
    accuracy: float
    predictions: Union[pd.Series, List[int]]
    tag_sentiment: Callable
    model: Model


class SentimentAnalyzer:
    LABELS = (-1, 0, 1)

    @section_printing_decorator("SENTIMENT ANALYSIS")
    def __init__(self, dataset: DataSet, config_file_path: str):
        self._dataset = dataset
        self._config_file_path = config_file_path
        self._lstm_model = None
        self._best_model_info = self.get_best_sentiment_analysis_model()

    def _generate_confusion_matrix(self, predicted_labels):
        confusion_matrix = sklearn.metrics.confusion_matrix(
            self._dataset.initial_df_with_emojis_converted_to_text["label"], predicted_labels, labels=self.LABELS
        )
        confusion_matrix_percentage = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        df_cm_percentage = pd.DataFrame(confusion_matrix_percentage, index=self.LABELS, columns=self.LABELS)

        sn.set_theme(font_scale=1.4)
        cmap = sn.color_palette("rocket_r", as_cmap=True)
        sn.heatmap(
            df_cm_percentage, annot=True, annot_kws={"size": 16}, cmap=cmap, fmt=".2%",
            xticklabels=["pr negative", "pr neutral", "pr positive"],
            yticklabels=["true negative", "true neutral", "true positive"]
        )
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _get_total_accuracy(original_labeled_df, predicted_labels):
        correct_predictions = [
            test_prediction == list(original_labeled_df[LABEL_COLUMN_NAME])[index]
            for index, test_prediction in enumerate(predicted_labels)
        ]
        accuracy = sum(correct_predictions) / len(correct_predictions)
        return accuracy


    def train_lstm(self, train_dataset: DataSet, main_test_dataset: DataSet):
        tokenizer = Tokenizer(num_words=1500)
        tokenizer.fit_on_texts(train_dataset.normalized_tweets_as_demojized_strings)
        train_sequences = tokenizer.texts_to_sequences(train_dataset.normalized_tweets_as_demojized_strings)
        main_sequences = tokenizer.texts_to_sequences(main_test_dataset.normalized_tweets_as_demojized_strings)
        padded_tweets = pad_sequences(train_sequences)
        padded_main_tweets = pad_sequences(main_sequences)
        y_train = to_categorical(train_dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME], num_classes=3)

        model = Sequential([
            layers.Embedding(1500, 100),
            layers.Bidirectional(layers.LSTM(100, dropout=0.5, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(9, dropout=0.2)),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(padded_tweets, y_train, epochs=20, use_multiprocessing=True,
                  callbacks=[EarlyStopping(monitor='accuracy', mode='max', min_delta=1, patience=2)]
                  )
        self._lstm_model = model
        transfer_predictions = model.predict(padded_main_tweets)
        transfer_predictions = pd.Series(
            [
                prediction if prediction != 2 else -1
                for prediction in np.argmax(transfer_predictions,axis=1)
            ]
        )
        return model, transfer_predictions


    def get_lstm_transfer_learning_results(self, convert_emojis_to_text=True):
        learning_data = DatasetLoader.load_transfer_learning_dataset(
            config_file_path=self._config_file_path, dataset=self._dataset, convert_emojis_to_text=convert_emojis_to_text
        )
        model, transfer_predictions = self.train_lstm(learning_data[0].train_data_nltk, self._dataset)
        accuracy = self._get_total_accuracy(self._dataset.initial_df_with_emojis_converted_to_text, transfer_predictions)
        return ModelInfo(
            accuracy=accuracy,
            predictions=transfer_predictions,
            tag_sentiment=model,
            model=Model.LSTM,
        )

    def get_lexicon_model_results(self):
        df_vader = Labeler.get_data_sentiment_with_vader(self._dataset.initial_df_with_emojis_converted_to_text)
        accuracy = self._get_total_accuracy(self._dataset.initial_df_with_emojis_converted_to_text, df_vader["label"])
        return ModelInfo(
            accuracy=accuracy,
            predictions=df_vader["label"],
            tag_sentiment=Labeler.get_data_sentiment_with_vader,
            model=Model.VADER,
        )

    def get_llm_results(self):
        sentiment_analysis = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")
        result = sentiment_analysis(self._dataset.normalized_tweets_as_demojized_strings)
        label_map = {"LABEL_0": -1,  "LABEL_1": 0, "LABEL_2": 1,}
        predictions = [label_map.get(res["label"]) for res in result]
        accuracy = self._get_total_accuracy(self._dataset.initial_df_with_emojis_converted_to_text, predictions)
        return ModelInfo(
            accuracy=accuracy,
            predictions=predictions,
            tag_sentiment=sentiment_analysis,
            model=Model.LLM,
        )

    def get_best_sentiment_analysis_model(self):
        lstm_info = self.get_lstm_transfer_learning_results()
        lexicon_info = self.get_lexicon_model_results()
        llm_info = self.get_llm_results()
        return max([lstm_info, lexicon_info, llm_info], key=lambda x: x.accuracy)

    def show_sentiment_labeling_accuracy(self):
        self._generate_confusion_matrix(self._best_model_info.predictions)

    def tag_dataset_with_sentiment(self, dataset: DataSet):
        predictions = self._best_model_info.tag_sentiment(dataset.normalized_tweets_as_demojized_strings)
        if self._best_model_info.model == Model.LLM:
            label_map = {"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1, }
            predictions = [label_map.get(res["label"]) for res in predictions]
        dataset.initial_df[LABEL_COLUMN_NAME] = predictions
        return dataset.initial_df
