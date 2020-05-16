import emoji
import pandas as pd
from langdetect import detect, lang_detect_exception

from data.column_names import LIV_WAT_TEXT_COLUMN_NAME


class TweetsFilterer:
    ENGLISH_LANGUAGE_LABEL = "en"
    RETWEET_INDICATOR = "RT @"
    RETWEET_INDICATOR_POSITIONS = slice(0, 4)

    @classmethod
    def filter_out_tweets_with_invalid_language_text(
        cls, df: pd.DataFrame
    ) -> pd.DataFrame:
        return df[df.apply(cls._tweet_contains_valid_text, axis=1)]

    @classmethod
    def filter_out_retweets(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df[~df.apply(cls._tweet_is_a_retweet, axis=1)]

    @classmethod
    def _tweet_contains_valid_text(cls, tweet: pd.Series) -> bool:
        text = cls._get_text_from_tweet(tweet)
        try:
            return cls._text_is_written_in_english(text)
        except lang_detect_exception.LangDetectException:
            return cls._text_contains_emojis(text)

    @classmethod
    def _text_is_written_in_english(cls, text: str) -> bool:
        detected_language = detect(text)
        if detected_language == cls.ENGLISH_LANGUAGE_LABEL:
            return True
        else:
            return False

    @classmethod
    def _text_contains_emojis(cls, text: str) -> bool:
        return True if emoji.emoji_count(text) else False

    @classmethod
    def _tweet_is_a_retweet(cls, tweet: pd.Series) -> bool:
        text = cls._get_text_from_tweet(tweet)
        return text[cls.RETWEET_INDICATOR_POSITIONS] == cls.RETWEET_INDICATOR

    @classmethod
    def _get_text_from_tweet(cls, tweet: pd.Series) -> str:
        return tweet[LIV_WAT_TEXT_COLUMN_NAME]
