from scripts.common import (
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_RELATIVE_FILE_PATH,
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_RELATIVE_FILE_PATH,
)

from configs.config_loader import ConfigLoader
from corpus.corpus_transformation_pipeline import CorpusTransformer
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from data.loaders import DataLoader
import pandas as pd


def save_df_with_normalized_tweets():
    configs = ConfigLoader.load()
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_RELATIVE_FILE_PATH
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    normalized_corpus = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets, hyper_parameters_config=configs.settings["setting_for_tweet_cleaning"],
    )
    normalized_corpus = [" ".join(tweet_words) for tweet_words in normalized_corpus]
    new_df = pd.DataFrame(normalized_corpus, columns=[LIV_WAT_TEXT_COLUMN_NAME])
    new_df.to_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_RELATIVE_FILE_PATH,
    )


if __name__ == "__main__":
    save_df_with_normalized_tweets()
