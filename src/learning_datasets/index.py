from typing import Optional, Dict, List
from dataclasses import dataclass
from data.utils import DataSet
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

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


TRANSFER_LEARNING_DATASETS_INFO = [
    TransferLearningDataSetInfo(
        path=os.path.join(BASE_PATH, "dataset_1.csv"),
        text_column_name="clean_text",
        label_column_name="category",
    ),
    TransferLearningDataSetInfo(
        path=os.path.join(BASE_PATH, "dataset_2.csv"),
        text_column_name="text",
        label_column_name="sentiment",
        sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    ),
    TransferLearningDataSetInfo(
        path=os.path.join(BASE_PATH, "dataset_3.csv"),
        text_column_name="Product_Description",
        label_column_name="Sentiment",
        sentiment_label_conversion_map={0: 0, 3: 0,  1: -1, 2: 1},
    ),
    TransferLearningDataSetInfo(
        path=os.path.join(BASE_PATH, "dataset_4.csv"),
        text_column_name="original_text",
        label_column_name="sentiment_class",
    ),
    # TransferLearningDataSetInfo(
    #     path="learning_datasets/ivankunchev_tweet-sentiment-extraction-ml/train_ml.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSetInfo(
    #     path="learning_datasets/louise2001_extended-train-for-tweet/extended_train.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSetInfo(
    #     path="learning_datasets/maxjon_complete-tweet-sentiment-extraction-data/tweet_dataset.csv",
    #     text_column_name="text",
    #     label_column_name="new_sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSetInfo(
    #     path="learning_datasets/seshurajup_tweet-sentiment-extraction-old/dataset_2.csv",
    #     text_column_name="text",
    #     label_column_name="sentiment",
    #     sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
    # ),
    # TransferLearningDataSetInfo(
    #     path="learning_datasets/vivekrathi055_sentiment-analysis-on-financial-tweets/tweet_sentiment.csv",
    #     text_column_name="cleaned_tweets",
    #     label_column_name="sentiment",
    # ),
]
