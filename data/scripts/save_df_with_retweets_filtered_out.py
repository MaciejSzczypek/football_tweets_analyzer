from data.scripts.common import (
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_RELATIVE_FILE_PATH,
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_RELATIVE_FILE_PATH,
)
from data.filters import TweetsFilterer
from data.loaders import DataLoader

if __name__ == "__main__":
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_RELATIVE_FILE_PATH
    )
    new_df = TweetsFilterer.filter_out_retweets(df=df).reset_index()
    new_df.to_csv(
        LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_RELATIVE_FILE_PATH,
        columns=df.columns.values,
    )
