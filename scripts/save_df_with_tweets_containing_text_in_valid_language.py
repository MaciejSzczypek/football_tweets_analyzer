from common import (
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_RELATIVE_FILE_PATH,
    LIVERPOOL_VS_WATFORD_ORIGINAL_RELATIVE_FILE_PATH,
)
from data.filters import TweetsFilterer
from data.loaders import DataLoader

if __name__ == "__main__":
    df = DataLoader.from_csv(LIVERPOOL_VS_WATFORD_ORIGINAL_RELATIVE_FILE_PATH)
    df = TweetsFilterer.filter_out_tweets_with_invalid_language_text(df=df)
    df.to_csv(LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_RELATIVE_FILE_PATH)
