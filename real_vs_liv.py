from configs.config_loader import ConfigLoader
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover
from data.loaders import DataLoader
from data.paths import (
    REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME,
    DataFilePaths,
    REAL_VS_LIVERPOOL_FILE_NAME_CORE,
)
import pandas as pd
from data.utils import DataSet
from information_extraction import (
    show_basic_facts,
    show_most_relevant_sentences,
    show_topics_modeled_with_nmf,
    show_top_ngrams,
    show_word_cloud,
    show_summaries_generated_with_transformers,
    show_time_frames_analysis,
)
from sentiment.analysis import show_sentiment_analysis_accuracies_results
from sklearn.feature_extraction.text import TfidfVectorizer
from information_extraction.enums import Season, League


def run_analysis():
    # data loading
    real_liverpool_file_paths = DataFilePaths(
        configuration_name=REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME,
        core_name=REAL_VS_LIVERPOOL_FILE_NAME_CORE,
    )
    configs = ConfigLoader.load(real_liverpool_file_paths.configuration_path)
    initial_df_with_tweet_specific_noise_removed = DataLoader.from_csv(
        real_liverpool_file_paths.tweet_specific_noise_removed, line_terminator="\n"
    )
    # initial_df_with_tweet_specific_noise_removed = initial_df_with_tweet_specific_noise_removed.sample(n=40_000, random_state=1)
    original_df = DataLoader.from_csv(real_liverpool_file_paths.original_path, line_terminator="\n")

    # data preparation
    df = DuplicatedTweetsRemover.remove_duplicated_tweets(
        initial_df_with_tweet_specific_noise_removed
    )
    dataset_without_emoticons = DataSet.create(
        df=df.copy(),
        hyper_parameters_config=configs.settings["analysis_without_emoticons"],
        original_df=original_df,
    )
    # dataset_with_emoticons = DataSet.create(
    #     df=df.copy(),
    #     hyper_parameters_config=configs.settings["analysis_with_emoticons"],
    # )
    # labeled_df_for_sentiment_analysis_tests = DataLoader.from_csv(real_liverpool_file_paths.random_batch_fully_tagged)
    # labeled_dataset_for_sentiment_analysis_tests = DataSet.create(
    #     df=labeled_df_for_sentiment_analysis_tests,
    #     hyper_parameters_config=configs.settings["setting_for_sentiment_analysis"],
    #     tfidf_vectorizer=TfidfVectorizer(token_pattern=r"\S+", stop_words="english")
    # )

    # top n-grams
    # show_top_ngrams(corpus=dataset_without_emoticons.normalized_tweets_as_token_lists)
    # show_word_cloud(text=dataset_without_emoticons.flat_text, normalize_plurals=False)

    # facts extraction
    # show_basic_facts(
    #     corpus_without_emoticons=dataset_without_emoticons.initial_tweets_array,
    #     corpus_with_emoticons=dataset_with_emoticons.initial_tweets_array,
    #     n_latest_tweets_used_for_result_collection=5000,
    #     season=Season.SEASON_17_18,
    #     league=League.CHAMPIONS_LEAGUE,
    # )
    # summarization
    # show_most_relevant_sentences(
    #     normalized_tweets_as_strings=dataset_without_emoticons.normalized_tweets_as_strings,
    #     df_before_transformation=dataset_without_emoticons.initial_df,
    #     sentences=dataset_without_emoticons.normalized_tweets_as_token_lists,
    #     random_batch_size=25_000,
    # )
    # show_summaries_generated_with_transformers(dataset_without_emoticons.df_with_normalized_tweets)
    # topic modeling
    # df_with_topic_labels = show_topics_modeled_with_nmf(
    #     dataset=dataset_without_emoticons,
    #     n_of_topics=7,
    # )
    # show_sentiment_analysis_accuracies_results(
    #     dataset=labeled_dataset_for_sentiment_analysis_tests,
    #     config_file_path=real_liverpool_file_paths.configuration_path,
    #     accuracies_df_path_to_load="/home/maciej_szczypek/PJATK/master_thesis/python_project/data/auxiliary_files/accuracies_df_real_liv.csv",
    #     # output_accuracies_df_path="/home/maciej_szczypek/PJATK/master_thesis/python_project/data/auxiliary_files/accuracies_df_real_liv.csv",
    # )
    # time frame analysis
    show_time_frames_analysis(
        # df_sentiment=dataset_with_emoticons.initial_df_with_emojis_converted_to_text,
        # df_topic=df_with_topic_labels,
        df_sentiment=pd.read_csv("results/df_sentiment"),
        df_topic=pd.read_csv("results/df_topic", lineterminator="\n"),
        time_frame_minutes_length=15
    )


if __name__ == "__main__":
    run_analysis()
