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


TRANSFER_LEARNING_DATASET_INFO = TransferLearningDataSetInfo(
    path=os.path.join(BASE_PATH, "tl_dataset.csv"),
    text_column_name="text",
    label_column_name="sentiment",
    sentiment_label_conversion_map={"positive": 1, "neutral": 0, "negative": -1},
)
