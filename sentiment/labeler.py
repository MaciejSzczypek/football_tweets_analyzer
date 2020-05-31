from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import List, Callable
import pandas as pd
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from sentiment.labels import SentimentLabel
from corpus.tokenizing.custom_tokenizers import CustomTokenizer
from nltk.corpus import sentiwordnet
from nltk import pos_tag


class Labeler:
    LABEL_COLUMN_NAME = "label"
    POLARITY_COLUMN_NAME = "polarity"

    @classmethod
    def get_data_sentiment_with_vader(cls, df: pd.DataFrame) -> pd.DataFrame:
        analyser = SentimentIntensityAnalyzer()
        labeled_df = cls._get_data_sentiment(
            df=df,
            sentiment_polarity_calculator=lambda text: analyser.polarity_scores(text)["compound"],
        )
        return labeled_df

    @classmethod
    def get_data_sentiment_with_text_blob(cls, df: pd.DataFrame) -> pd.DataFrame:
        analyser = TextBlob
        labeled_df = cls._get_data_sentiment(
            df=df,
            sentiment_polarity_calculator=lambda text: analyser(text).sentiment.polarity,
        )
        return labeled_df

    @classmethod
    def get_data_sentiment_with_sentiwordnet(cls, df: pd.DataFrame) -> pd.DataFrame:
        analyser = cls._sentiwordnet_sentiment_analyser
        labeled_df = cls._get_data_sentiment(
            df=df,
            sentiment_polarity_calculator=analyser,
        )
        return labeled_df

    @classmethod
    def _get_data_sentiment(
        cls,
        df: pd.DataFrame,
        sentiment_polarity_calculator: Callable
    ) -> pd.DataFrame:
        labeled_df = pd.DataFrame()
        labeled_df[cls.POLARITY_COLUMN_NAME] = (
            df[LIV_WAT_TEXT_COLUMN_NAME].apply(sentiment_polarity_calculator)
        )
        labeled_df[cls.LABEL_COLUMN_NAME] = (
            labeled_df[cls.POLARITY_COLUMN_NAME].apply(cls._get_label_from_sentiment_score)
        )
        return labeled_df

    @classmethod
    def _get_label_from_sentiment_score(cls, sentiment_score: int) -> int:
        if sentiment_score >= 0.05:
            return SentimentLabel.positive.value
        elif sentiment_score <= -0.05:
            return SentimentLabel.negative.value
        else:
            return SentimentLabel.neutral.value

    @classmethod
    def _sentiwordnet_sentiment_analyser(cls, text: str):
        tokenized_text = CustomTokenizer.tokenize(text)
        tagged_tokens = pos_tag(tokenized_text)
        pos_score = 0
        neg_score = 0
        token_count = 0
        obj_score = 0
        for token, tag in tagged_tokens:
            ss_set = None
            if "NN" in tag and list(sentiwordnet.senti_synsets(token, "n")):
                ss_set = list(sentiwordnet.senti_synsets(token, "n"))[0]
            elif "VB" in tag and list(sentiwordnet.senti_synsets(token, "v")):
                ss_set = list(sentiwordnet.senti_synsets(token, "v"))[0]
            elif "JJ" in tag and list(sentiwordnet.senti_synsets(token, "a")):
                ss_set = list(sentiwordnet.senti_synsets(token, "a"))[0]
            elif "RB" in tag and list(sentiwordnet.senti_synsets(token, "r")):
                ss_set = list(sentiwordnet.senti_synsets(token, "r"))[0]
            if ss_set:
                pos_score += ss_set.pos_score()
                neg_score += ss_set.neg_score()
                obj_score += ss_set.obj_score()
                token_count += 1
        final_score = pos_score - neg_score
        normalized_final_score = final_score / token_count if token_count else 0
        return normalized_final_score
