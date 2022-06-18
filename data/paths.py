import os
from dataclasses import dataclass

# region common
DATA_DIR_NAME = "data"
DATA_FILES_DIR_NAME = "source_files"
FINAL_FILES_DIR_NAME = "final_files"
CONFIG_FILES_DIR_NAME = "configs"
CONFIG_FILES_SUB_DIR_NAME = "files"
ABS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILES_DIR_NAME)
ABS_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_FILES_DIR_NAME,CONFIG_FILES_SUB_DIR_NAME)

@dataclass
class DataFilePaths:
    configuration_name: str
    core_name: str

    def _create_path(self, suffix: str):
        return os.path.join(ABS_DATA_DIR, f"{self.core_name}_{suffix}")

    @property
    def configuration_path(self):
        return os.path.join(ABS_CONFIG_DIR, self.configuration_name)

    @property
    def original_path(self):
        return self._create_path("original")

    @property
    def english_tweets(self):
        return self._create_path("english_tweets")

    @property
    def english_tweets_with_retweets_removed(self):
        return self._create_path("english_tweets_with_retweets_removed")

    @property
    def tweet_specific_noise_removed(self):
        return self._create_path("with_tweet_specific_noise_removed")

# endregion common

# region liverpool watford
LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_NAME = "liv_vs_wat.yml"
LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_PATH = os.path.join(
    CONFIG_FILES_DIR_NAME, CONFIG_FILES_SUB_DIR_NAME, LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_NAME
)
LIVERPOOL_VS_WATFORD_ORIGINAL_FILE_NAME = "liverpool_vs_watford_original"
LIVERPOOL_VS_WATFORD_ORIGINAL_FILE_PATH = os.path.join(
    DATA_DIR_NAME, DATA_FILES_DIR_NAME, LIVERPOOL_VS_WATFORD_ORIGINAL_FILE_NAME
)
LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_FILE_NAME = (
    "liverpool_vs_watford_english_tweets"
)
LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_FILE_PATH = os.path.join(
    DATA_DIR_NAME,
    DATA_FILES_DIR_NAME,
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_FILE_NAME,
)
LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_NAME = (
    "liverpool_vs_watford_english_tweets_with_retweets_removed"
)
LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_PATH = os.path.join(
    DATA_DIR_NAME,
    DATA_FILES_DIR_NAME,
    LIVERPOOL_VS_WATFORD_WITH_NON_ENGLISH_TWEETS_EXCLUDED_AND_RETWEETS_REMOVED_FILE_NAME,
)
LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_NAME = (
    "liverpool_vs_watford_with_tweet_specific_noise_removed"
)
LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH = os.path.join(
    DATA_DIR_NAME,
    DATA_FILES_DIR_NAME,
    LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_NAME,
)

LIVERPOOL_VS_WATFORD_LABELED_FILE_NAME = "liverpool_vs_watford_labeled"
LIVERPOOL_VS_WATFORD_LABELED_FILE_PATH = os.path.join(
    DATA_DIR_NAME, DATA_FILES_DIR_NAME, LIVERPOOL_VS_WATFORD_LABELED_FILE_NAME,
)
LIVERPOOL_VS_WATFORD_RANDOMLY_SELECTED_FILE_NAME = "liverpool_vs_watford_random_batch"
LIVERPOOL_VS_WATFORD_RANDOMLY_SELECTED_FILE_PATH = os.path.join(
    DATA_DIR_NAME,
    DATA_FILES_DIR_NAME,
    LIVERPOOL_VS_WATFORD_RANDOMLY_SELECTED_FILE_NAME,
)
LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_NAME = (
    "liverpool_vs_watford_test_batch_fully_tagged")
LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_PATH = os.path.join(
    DATA_DIR_NAME,
    FINAL_FILES_DIR_NAME,
    LIVERPOOL_VS_WATFORD_RANDOM_BATCH_FULLY_TAGGED_FILE_NAME,
)
# endregion liverpool watford
# region real liverpool
REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME = "real_vs_liv.yml"
REAL_VS_LIVERPOOL_FILE_NAME_CORE = "real_vs_liverpool"
# endregion real liverpool


# region auxiliary files
AUXILIARY_FILES_SUB_DIR_NAME = "auxiliary_files"
ENGLISH_CLUBS_FILE_NAME = "english_clubs.csv"
SPANISH_CLUBS_FILE_NAME = "spanish_clubs.csv"
ENGLISH_CLUBS_FILE_PATH = os.path.join(DATA_DIR_NAME, AUXILIARY_FILES_SUB_DIR_NAME, ENGLISH_CLUBS_FILE_NAME)
SPANISH_CLUBS_FILE_PATH = os.path.join(DATA_DIR_NAME, AUXILIARY_FILES_SUB_DIR_NAME, SPANISH_CLUBS_FILE_NAME)
# endregion auxiliary files
