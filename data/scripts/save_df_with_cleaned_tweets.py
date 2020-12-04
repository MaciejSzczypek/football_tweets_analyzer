from configs.config_loader import ConfigLoader
from corpus.corpus_transformation_pipeline import CorpusTransformer
from data.column_names import TEXT_COLUMN_NAME
from data.loaders import DataLoader
from data.paths import (
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH,
)


def save_df_with_normalized_tweets():
    configs = ConfigLoader.load()
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_PATH
    )
    df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].fillna("")
    tweets = df[TEXT_COLUMN_NAME].to_numpy()
    normalized_corpus = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets,
        hyper_parameters_config=configs.settings["setting_for_tweet_pre_cleaning"],
        remove_empty_tweets=False,
    )
    normalized_corpus = [
        " ".join(tweet_words) for tweet_words in normalized_corpus.corpus
    ]
    new_df = df.copy()
    new_df[TEXT_COLUMN_NAME] = normalized_corpus
    new_df = new_df[new_df["text"].astype(bool)]
    new_df.to_csv(LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH)
    print(len(new_df))


if __name__ == "__main__":
    save_df_with_normalized_tweets()
