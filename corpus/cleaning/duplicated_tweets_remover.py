import pandas as pd
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME, LIV_WAT_USER_NAME_COLUMN_NAME


class DuplicatedTweetsRemover:
    MINIMUM_NUMBER_OF_WORDS_TO_CONSIDER_TWEET_AS_DUPLICATE = 7

    @classmethod
    def remove_duplicated_tweets(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._remove_duplicated_tweets_from_the_same_user(df)
        df = cls._remove_most_likely_duplicated_tweets_without_rt_marker(df)
        return df

    @classmethod
    def _remove_duplicated_tweets_from_the_same_user(cls,  df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(
            [LIV_WAT_TEXT_COLUMN_NAME, LIV_WAT_USER_NAME_COLUMN_NAME],
            keep="first",
        )

    @classmethod
    def _remove_most_likely_duplicated_tweets_without_rt_marker(
            cls, df: pd.DataFrame
    ) -> pd.DataFrame:
        possible_duplicates = df[df.duplicated(LIV_WAT_TEXT_COLUMN_NAME)]
        possible_duplicates_check_if_they_can_be_considered_as_duplicates = (
            possible_duplicates.apply(
                cls._tweet_can_be_considered_as_retweet,
                axis=1,
            )
        )
        duplicates = (
            possible_duplicates[possible_duplicates_check_if_they_can_be_considered_as_duplicates]
        )
        return df[
            ~df.index.isin(duplicates.index)
        ]

    @classmethod
    def _tweet_can_be_considered_as_retweet(cls, row: pd.Series):
        return (
            len(row[LIV_WAT_TEXT_COLUMN_NAME].split())
            >= cls.MINIMUM_NUMBER_OF_WORDS_TO_CONSIDER_TWEET_AS_DUPLICATE
        )
