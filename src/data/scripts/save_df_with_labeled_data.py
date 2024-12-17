import pandas as pd

from data.loaders import DataLoader
from data.paths import (
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_LABELED_FILE_PATH,
)
from sentiment.labeler import Labeler

VADER_LABEL_COLUMN_NAME = "VaderLabel"
VADER_POLARITY_COLUMN_NAME = "VaderPolarity"
TEXTBLOB_LABEL_COLUMN_NAME = "TextBlobLabel"
TEXTBLOB_POLARITY_COLUMN_NAME = "TextBlobPolarity"
SENTIWORDNET_LABEL_COLUMN_NAME = "SentiWordNetLabel"
SENTIWORDNET_POLARITY_COLUMN_NAME = "SentiWordNetPolarity"


def save_df_with_labeled_tweets():
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    df_vader_sentiment_result = Labeler.get_data_sentiment_with_vader(df)
    df_textblob_sentiment_result = Labeler.get_data_sentiment_with_text_blob(df)
    df_senti_word_net_sentiment_result = Labeler.get_data_sentiment_with_senti_word_net(df)
    aggregated_df = pd.DataFrame(
        {
            **df.to_dict(),
            VADER_POLARITY_COLUMN_NAME: df_vader_sentiment_result[
                Labeler.POLARITY_COLUMN_NAME
            ],
            VADER_LABEL_COLUMN_NAME: df_vader_sentiment_result[
                Labeler.LABEL_COLUMN_NAME
            ],
            TEXTBLOB_POLARITY_COLUMN_NAME: df_textblob_sentiment_result[
                Labeler.POLARITY_COLUMN_NAME
            ],
            TEXTBLOB_LABEL_COLUMN_NAME: df_textblob_sentiment_result[
                Labeler.LABEL_COLUMN_NAME
            ],
            SENTIWORDNET_POLARITY_COLUMN_NAME: df_senti_word_net_sentiment_result[
                Labeler.POLARITY_COLUMN_NAME
            ],
            SENTIWORDNET_LABEL_COLUMN_NAME: df_senti_word_net_sentiment_result[
                Labeler.LABEL_COLUMN_NAME
            ],
        }
    )
    aggregated_df.to_csv(LIVERPOOL_VS_WATFORD_LABELED_FILE_PATH)


if __name__ == "__main__":
    save_df_with_labeled_tweets()
