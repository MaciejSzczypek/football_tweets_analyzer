import os
from dataclasses import dataclass
from pathlib import Path


# region common
DATA_DIR_NAME = "data"
PROJECT_LEVEL_PATH = str(Path(__file__).parents[2])
ABS_SOURCE_FILES_DIR = os.path.join(PROJECT_LEVEL_PATH, DATA_DIR_NAME, "source_files")
ABS_AUXILIARY_FILES_DIR = os.path.join(PROJECT_LEVEL_PATH, DATA_DIR_NAME, "auxiliary_files")
ABS_CONFIG_DIR = os.path.join(PROJECT_LEVEL_PATH, "configs")


@dataclass
class DataFilePaths:
    configuration_name: str
    core_name: str

    def _create_path(self, suffix: str):
        return os.path.join(ABS_SOURCE_FILES_DIR, f"{self.core_name}_{suffix}")

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

    @property
    def random_batch(self):
        return self._create_path("random_batch")

    @property
    def random_batch_fully_tagged(self):
        return self._create_path("random_batch_fully_tagged")

# endregion common

# region liverpool watford
LIVERPOOL_VS_WATFORD_CONFIGURATION_FILE_NAME = "liv_vs_wat.yml"
LIVERPOOL_VS_WATFORD_FILE_NAME_CORE = "liverpool_vs_watford"
# endregion liverpool watford
# region real liverpool
REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME = "real_vs_liv.yml"
REAL_VS_LIVERPOOL_FILE_NAME_CORE = "real_vs_liverpool"
# endregion real liverpool


# region auxiliary files
ENGLISH_CLUBS_FILE_NAME = "english_clubs.csv"
SPANISH_CLUBS_FILE_NAME = "spanish_clubs.csv"
ENGLISH_CLUBS_FILE_PATH = os.path.join(ABS_AUXILIARY_FILES_DIR, ENGLISH_CLUBS_FILE_NAME)
SPANISH_CLUBS_FILE_PATH = os.path.join(ABS_AUXILIARY_FILES_DIR, SPANISH_CLUBS_FILE_NAME)
# endregion auxiliary files
