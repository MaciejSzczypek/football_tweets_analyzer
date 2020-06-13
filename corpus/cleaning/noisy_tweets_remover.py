import pandas as pd
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME, LIV_WAT_USER_NAME_COLUMN_NAME


class NoisyTweetsRemover:
    NUMBER_OF_WORDS_THRESHOLD_TO_CONSIDER_TWEET_AS_DUPLICATE = 9

    @classmethod
    def remove_noisy_tweets(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._remove_duplicated_tweets_from_the_same_user(df)
        df = cls._remove_possible_retweets_without_rt_marker(df)
        return df

    @classmethod
    def _remove_duplicated_tweets_from_the_same_user(cls,  df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(
            [LIV_WAT_TEXT_COLUMN_NAME, LIV_WAT_USER_NAME_COLUMN_NAME],
            keep="first",
        )

    @classmethod
    def _remove_possible_retweets_without_rt_marker(cls,  df: pd.DataFrame) -> pd.DataFrame:
        duplicates = df[df.duplicated(LIV_WAT_TEXT_COLUMN_NAME)]
        duplicates_with_information_if_they_can_be_considered_as_retweet = (
            duplicates.apply(
                cls._tweet_can_be_considered_as_retweet,
                axis=1,
            )
        )
        duplicates_to_be_considered_as_retweet = (
            duplicates[~duplicates_with_information_if_they_can_be_considered_as_retweet]
        )
        first_occurrences_of_duplicate_considered_as_retweet = (
            duplicates_to_be_considered_as_retweet.drop_duplicates(
                LIV_WAT_TEXT_COLUMN_NAME,
                keep="first"
            )
        )
        duplicates_to_drop = duplicates_to_be_considered_as_retweet[
            ~duplicates_to_be_considered_as_retweet.index.isin(
                first_occurrences_of_duplicate_considered_as_retweet.index
            )
        ]
        return df[
            ~df.index.isin(duplicates_to_drop.index)
        ]

    @classmethod
    def _tweet_can_be_considered_as_retweet(cls, row: pd.Series):
        return (
            len(row[LIV_WAT_TEXT_COLUMN_NAME].split())
            > cls.NUMBER_OF_WORDS_THRESHOLD_TO_CONSIDER_TWEET_AS_DUPLICATE
        )