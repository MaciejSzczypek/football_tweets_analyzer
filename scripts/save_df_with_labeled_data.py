from data.loaders import DataLoader
from scripts.common import (
    LIVERPOOL_VS_WATFORD_LABELED_RELATIVE_FILE_PATH,
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_RELATIVE_FILE_PATH,
)
from sentiment.labeler import Labeler


def save_df_with_labeled_tweets():
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_RELATIVE_FILE_PATH
    )
    Labeler.label_data_with_vader(df)


if __name__ == "__main__":
    save_df_with_labeled_tweets()
