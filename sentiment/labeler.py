from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import List, Callable
import pandas as pd
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from sentiment.labels import SentimentLabel


class Labeler:
    @classmethod
    def label_data_with_vader(cls, df: pd.DataFrame) -> pd.DataFrame:
        analyser = SentimentIntensityAnalyzer()
        labeled_df = cls._label_data(
            df=df,
            sentiment_analyser=lambda text: analyser.polarity_scores(text)["compound"],
        )
        print(labeled_df)
        return df

    @classmethod
    def label_data_with_text_blob(cls, df: pd.DataFrame) -> List[int]:
        pass

    @classmethod
    def _label_data(
        cls, df: pd.DataFrame, sentiment_analyser: Callable
    ) -> pd.DataFrame:
        labeled_df = df[LIV_WAT_TEXT_COLUMN_NAME].apply(sentiment_analyser)
        labeled_df = labeled_df.apply(cls._get_label_from_sentiment_score)
        return labeled_df

    @classmethod
    def _get_label_from_sentiment_score(cls, sentiment_score: int) -> int:
        if sentiment_score >= 0.05:
            return SentimentLabel.positive.value
        elif sentiment_score <= -0.05:
            return SentimentLabel.negative.value
        else:
            return SentimentLabel.neutral.value
