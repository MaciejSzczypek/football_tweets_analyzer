import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from configs.config_loader import ConfigLoader
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover
from data.loaders import DataLoader
from data.paths import (
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH,
    LIVERPOOL_VS_WATFORD_ORIGINAL_FILE_PATH,
    LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_PATH,
)
from data.utils import DataSet
from information_extraction import (
    show_basic_facts,
    show_most_relevant_sentences,
    show_summaries_generated_with_transformers,
    show_topics_modeled_with_nmf,
    show_top_ngrams,
    show_time_frames_analysis,
    show_word_cloud,
)
from information_extraction.enums import Season, League
from sentiment.analysis import show_sentiment_analysis_accuracies_results


def run_liverpool_watford_analysis():
    # data loading
    configs = ConfigLoader.load(LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_PATH)
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    original_df = DataLoader.from_csv(LIVERPOOL_VS_WATFORD_ORIGINAL_FILE_PATH)
    # labeled_df_for_sentiment_analysis_tests = DataLoader.from_csv(
    #     LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH
    # )
    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    dataset_without_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=configs.settings["analysis_without_emoticons"],
        original_df=original_df,
    )
    dataset_with_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=configs.settings["analysis_with_emoticons"],
    )
    # labeled_dataset_for_sentiment_analysis_tests = DataSet.create(
    #     df=labeled_df_for_sentiment_analysis_tests,
    #     hyper_parameters_config=configs.settings["setting_for_sentiment_analysis"],
    #     tfidf_vectorizer=TfidfVectorizer(token_pattern=r"\S+", stop_words="english")
    # )

    # top n-grams
    # show_top_ngrams(corpus=dataset_without_emoticons.normalized_tweets_as_token_lists)
    # show_word_cloud(text=dataset_without_emoticons.flat_text)

    # facts extraction
    # show_basic_facts(
    #     corpus_without_emoticons=dataset_without_emoticons.initial_tweets_array,
    #     corpus_with_emoticons=dataset_with_emoticons.initial_tweets_array,
    #     season=Season.SEASON_19_20,
    #     league=League.PREMIER_LEAGUE,
    # )
    # summarization
    # show_most_relevant_sentences(
    #     normalized_tweets_as_strings=dataset_without_emoticons.normalized_tweets_as_strings,
    #     df_before_transformation=dataset_without_emoticons.initial_df,
    #     sentences=dataset_without_emoticons.normalized_tweets_as_token_lists,
    # )
    # show_summaries_generated_with_transformers(dataset_without_emoticons.df_with_normalized_tweets)
    # topic modeling
    df_with_topic_labels = show_topics_modeled_with_nmf(
        dataset=dataset_without_emoticons,
    )
    # sentiment analysis
    # show_sentiment_analysis_accuracies_results(
    #     dataset=labeled_dataset_for_sentiment_analysis_tests,
    #     config_file_path=LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_PATH,
    #     accuracies_df_path_to_load="/home/maciej_szczypek/PJATK/master_thesis/python_project/data/auxiliary_files/accuracies_df.csv",
    #     # output_accuracies_df_path="/home/maciej_szczypek/PJATK/master_thesis/python_project/data/auxiliary_files/accuracies_df.csv",
    # )

    # time frame analysis
    show_time_frames_analysis(
        df_sentiment=dataset_with_emoticons.initial_df_with_emojis_converted_to_text,
        df_topic=df_with_topic_labels,
        # df_sentiment=pd.read_csv("results/df_sentiment"),
        # df_topic=pd.read_csv("results/df_topic"),
    )


if __name__ == "__main__":
    run_liverpool_watford_analysis()
