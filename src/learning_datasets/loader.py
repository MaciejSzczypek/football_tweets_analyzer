from typing import Dict, Tuple, Any, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from configs.config_loader import ConfigLoader
from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME
from data.utils import DataSet
from learning_datasets.index import TransferLearningDataSet, TRANSFER_LEARNING_DATASET_INFO


class DatasetLoader:

    @staticmethod
    def _load_datasets(
        file_path: str,
        text_column_name: str,
        label_column_name: str,
        sentiment_label_conversion_map: Optional[Dict[Any, int]],
        config_file_path: str,
    ) -> Tuple[DataSet, DataSet]:
        configs = ConfigLoader.load(config_file_path)
        print(file_path)
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

    @staticmethod
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
            dataset.tfidf.toarray(),  # [:,:,None],
            dataset.df_with_normalized_tweets[LABEL_COLUMN_NAME],
            test_size=0.1,
        )
        if not convert_emojis_to_text:
            x_main = dataset.tfidf_vectorizer.fit_transform(main_test_dataset.normalized_tweets_as_strings)
        else:
            x_main = dataset.tfidf_vectorizer.fit_transform(
                main_test_dataset.normalized_tweets_as_demojized_strings
            ).toarray()  # [:,:,None]
        return x_train, x_test, x_main, y_train, y_test

    @classmethod
    def load_transfer_learning_dataset(cls, config_file_path: str, dataset: DataSet, convert_emojis_to_text):
        datasets = []
        train_dataset, test_dataset = cls._load_datasets(
            file_path=TRANSFER_LEARNING_DATASET_INFO.path,
            text_column_name=TRANSFER_LEARNING_DATASET_INFO.text_column_name,
            label_column_name=TRANSFER_LEARNING_DATASET_INFO.label_column_name,
            sentiment_label_conversion_map=TRANSFER_LEARNING_DATASET_INFO.sentiment_label_conversion_map,
            config_file_path=config_file_path,
        )
        x_train, x_test, x_main, y_train, y_test = cls._load_tfidf_datasets(
            file_path=TRANSFER_LEARNING_DATASET_INFO.path,
            text_column_name=TRANSFER_LEARNING_DATASET_INFO.text_column_name,
            label_column_name=TRANSFER_LEARNING_DATASET_INFO.label_column_name,
            main_test_dataset=dataset,
            sentiment_label_conversion_map=TRANSFER_LEARNING_DATASET_INFO.sentiment_label_conversion_map,
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
        datasets.append(tl_dataset)
        return datasets