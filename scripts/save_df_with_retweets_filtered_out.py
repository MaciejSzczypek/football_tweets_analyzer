from data.filters import TweetsFilterer
from data.loaders import DataLoader

from common import (
    RELATIVE_LANGUAGE_FILTERED_TWEETS_LIVERPOOL_VS_WATFORD_FILE_PATH,
    RELATIVE_FINAL_LIVERPOOL_VS_WATFORD_WITH_RETWEETS_EXCLUDED_FILE_PATH,
)

if __name__ == "__main__":
    df = DataLoader.from_csv(
        RELATIVE_LANGUAGE_FILTERED_TWEETS_LIVERPOOL_VS_WATFORD_FILE_PATH
    )
    df = TweetsFilterer.filter_out_retweets(df=df)
    df.to_csv(RELATIVE_FINAL_LIVERPOOL_VS_WATFORD_WITH_RETWEETS_EXCLUDED_FILE_PATH)
