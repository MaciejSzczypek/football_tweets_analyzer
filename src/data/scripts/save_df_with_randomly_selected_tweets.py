from data.loaders import DataLoader
from data.paths import DataFilePaths, REAL_VS_LIVERPOOL_FILE_NAME_CORE, REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME
from data.column_names import TEXT_COLUMN_NAME

BATCH_SIZE = 400

COLUMNS_TO_SELECT = [TEXT_COLUMN_NAME]


def save_df_with_randomly_selected_tweets(data_file_paths: DataFilePaths):
    df = DataLoader.from_csv(data_file_paths.english_tweets_with_retweets_removed, line_terminator="\n")[COLUMNS_TO_SELECT]
    randomly_selected_tweets = df.sample(BATCH_SIZE)
    randomly_selected_tweets.to_csv(data_file_paths.random_batch)

    return df


if __name__ == "__main__":
    real_liverpool_file_paths = DataFilePaths(
        configuration_name=REAL_VS_LIVERPOOL_CONFIGURATION_FILE_NAME,
        core_name=REAL_VS_LIVERPOOL_FILE_NAME_CORE,
    )
    save_df_with_randomly_selected_tweets(real_liverpool_file_paths)
