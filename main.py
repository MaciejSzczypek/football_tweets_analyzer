from data.loaders import DataLoader
from data.paths import FINAL_LIVERPOOL_VS_WATFORD_WITH_RETWEETS_EXCLUDED_FILE_PATH
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from corpus.corpus_transformation_pipeline import CorpusTransformationPipeline
from configs.config_loader import ConfigLoader


def run_liverpool_watford_processing():
    config = ConfigLoader.load()
    df = DataLoader.from_csv(
        FINAL_LIVERPOOL_VS_WATFORD_WITH_RETWEETS_EXCLUDED_FILE_PATH
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    tokenized_and_normalized_corpus = CorpusTransformationPipeline.transform_twitter_corpus(
        corpus=tweets,
        config=config,
    )
    for x in tokenized_and_normalized_corpus:
        print(x)


if __name__ == "__main__":
    run_liverpool_watford_processing()
