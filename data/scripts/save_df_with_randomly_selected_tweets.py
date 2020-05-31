from data.loaders import DataLoader
from data.scripts.common import (
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_RELATIVE_FILE_PATH,
    LIVERPOOL_VS_WATFORD_RANDOMLY_SELECTED_RELATIVE_FILE_PATH,

)
from data.column_names import (
    LIV_WAT_TEXT_COLUMN_NAME
)
BATCH_SIZE = 400

COLUMNS_TO_SELECT = [
    LIV_WAT_TEXT_COLUMN_NAME
]


def save_df_with_randomly_selected_tweets():
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_RELATIVE_FILE_PATH,
    )[COLUMNS_TO_SELECT]
    randomly_selected_tweets = df.sample(BATCH_SIZE)
    randomly_selected_tweets.to_csv(LIVERPOOL_VS_WATFORD_RANDOMLY_SELECTED_RELATIVE_FILE_PATH)

    return df


if __name__ == "__main__":
    save_df_with_randomly_selected_tweets()
