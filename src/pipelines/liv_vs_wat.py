from sklearn.feature_extraction.text import TfidfVectorizer

from configs.config_loader import ConfigLoader
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover
from data.loaders import DataLoader
from data.utils import DataSet
from information_extraction.enums import Season, League
from information_extraction.facts_extractor import print_basic_facts
from information_extraction.keyphrase_extraction import KeyPhraseExtractor
from information_extraction.text_summarization import (
    print_most_relevant_sentences
)
from utils.file import make_directory_if_not_exists
from information_extraction.time_frames import show_time_frames_analysis
from information_extraction.topic_modelling import show_topics_modeled_with_nmf
from information_extraction.wordcloud import generate_word_cloud
from sentiment.analysis import SentimentAnalyzer
from data.paths import (
    DataFilePaths,
    LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_NAME,
    LIVERPOOL_VS_WATFORD_FILE_NAME_CORE,
)


def _prepare_datasets(file_paths, config):
    # data loading
    make_directory_if_not_exists(config.paths.results_dir)
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        file_paths.tweet_specific_noise_removed
    )
    original_df = DataLoader.from_csv(file_paths.original_path)
    labeled_df_for_sentiment_analysis_tests = DataLoader.from_csv(
        file_paths.random_batch_fully_tagged
    )
    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    dataset_without_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=config.settings["analysis_without_emoticons"],
        original_df=original_df,
    )
    dataset_with_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=config.settings["analysis_with_emoticons"],
    )
    labeled_dataset_for_sentiment_analysis_tests = DataSet.create(
        df=labeled_df_for_sentiment_analysis_tests,
        hyper_parameters_config=config.settings["setting_for_sentiment_analysis"],
        tfidf_vectorizer=TfidfVectorizer(token_pattern=r"\S+", stop_words="english")
    )
    return dataset_without_emoticons, dataset_with_emoticons, labeled_dataset_for_sentiment_analysis_tests


def run_liverpool_watford_analysis():
    # prepare data paths
    liverpool_watford_file_paths = DataFilePaths(
        configuration_name=LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_NAME,
        core_name=LIVERPOOL_VS_WATFORD_FILE_NAME_CORE,
    )
    # load configuration
    configs = ConfigLoader.load(liverpool_watford_file_paths.configuration_path)
    paths_config, sections_config = configs.paths, configs.sections

    # prepare data
    dataset_without_emoticons, dataset_with_emoticons, labeled_dataset_for_sentiment_analysis_tests = (
        _prepare_datasets(liverpool_watford_file_paths, configs)
    )

    # top n-grams
    if sections_config.is_top_n_grams_enabled:
        KeyPhraseExtractor.print_top_ngrams(corpus=dataset_without_emoticons.normalized_tweets_as_token_lists)

    # wordcloud
    if sections_config.is_wordcloud_enabled:
        generate_word_cloud(text=dataset_without_emoticons.flat_text, paths_config=paths_config)

    # facts extraction
    if sections_config.is_basic_facts_enabled:
        print_basic_facts(
            corpus_without_emoticons=dataset_without_emoticons.initial_tweets_array,
            corpus_with_emoticons=dataset_with_emoticons.initial_tweets_array,
            season=Season.SEASON_19_20,
            league=League.PREMIER_LEAGUE,
            paths_config=paths_config
        )

    # summarization
    if sections_config.is_summarization_enabled:
        print_most_relevant_sentences(
            normalized_tweets_as_strings=dataset_without_emoticons.normalized_tweets_as_strings,
            df_before_transformation=dataset_without_emoticons.initial_df,
            sentences=dataset_without_emoticons.normalized_tweets_as_token_lists,
        )

    # topic modeling
    if sections_config.is_topic_modeling_enabled:
        df_with_topic_labels = show_topics_modeled_with_nmf(
            paths_config=paths_config, dataset=dataset_without_emoticons
        )
    else:
        df_with_topic_labels = None

    # sentiment analysis
    if sections_config.is_sentiment_analysis_enabled:
        analyzer = SentimentAnalyzer(
            labeled_dataset_for_sentiment_analysis_tests, liverpool_watford_file_paths.configuration_path
        )
        analyzer.show_sentiment_labeling_accuracy()
        df_with_sentiment_labels = analyzer.tag_dataset_with_sentiment(dataset_with_emoticons)
    else:
        df_with_sentiment_labels = None

    # time frame analysis
    if sections_config.is_time_frames_enabled:
        show_time_frames_analysis(
            df_sentiment=df_with_sentiment_labels,
            df_topic=df_with_topic_labels,
            paths_config=paths_config
        )


if __name__ == "__main__":
    run_liverpool_watford_analysis()
