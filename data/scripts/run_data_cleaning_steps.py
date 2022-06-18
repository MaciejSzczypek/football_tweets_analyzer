from argparse import ArgumentParser

from data.filters import TweetsFilterer
from data.loaders import DataLoader
from data.paths import DataFilePaths, REAL_VS_LIVERPOOL_FILE_NAME_CORE, REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME
from data.column_names import TEXT_COLUMN_NAME
from configs.config_loader import ConfigLoader
from corpus.corpus_transformation_pipeline import CorpusTransformer


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("input_file_path", type=str)
    return parser


def remove_non_english_tweets(input_file_path: str, output_file_path: str, language_column_name: str = None):
    df = DataLoader.from_csv(input_file_path, line_terminator="\n")
    df = TweetsFilterer.filter_out_tweets_with_invalid_language_text(df=df)
    df.to_csv(output_file_path)

def remove_retweets(input_file_path: str, output_file_path: str):
    df = DataLoader.from_csv(input_file_path, line_terminator="\n")
    df = TweetsFilterer.filter_out_retweets(df=df).reset_index()
    df.to_csv(output_file_path, columns=df.columns.values)

def remove_tweet_speific_noise(config_file_path: str, input_file_path: str, output_file_path: str):
    configs = ConfigLoader.load(config_file_path)
    df = DataLoader.from_csv(input_file_path, line_terminator="\n")
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
    new_df = new_df[new_df[TEXT_COLUMN_NAME].astype(bool)]
    new_df.to_csv(output_file_path)


def run_data_cleaning(data_file_paths: DataFilePaths):
    # remove_non_english_tweets(data_file_paths.original_path, output_file_path=data_file_paths.english_tweets)
    remove_retweets(
        input_file_path=data_file_paths.english_tweets,
        output_file_path=data_file_paths.english_tweets_with_retweets_removed
    )
    remove_tweet_speific_noise(
        config_file_path=data_file_paths.configuration_path,
        input_file_path=data_file_paths.english_tweets_with_retweets_removed,
        output_file_path=data_file_paths.tweet_specific_noise_removed,
    )


if __name__ == "__main__":
    # real liverpool
    real_liverpool_file_paths = DataFilePaths(
        configuration_name=REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME,
        core_name=REAL_VS_LIVERPOOL_FILE_NAME_CORE,
    )
    run_data_cleaning(real_liverpool_file_paths)
