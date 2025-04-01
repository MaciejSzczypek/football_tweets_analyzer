import os.path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from typing import List

from configs.config_schema import PathsConfig
from data.column_names import CREATED_AT_COLUMN_NAME
from data.dataset_manager import DataSet
from information_extraction.keyphrase_extraction import KeyPhraseExtractor
from utils.printing import section_printing_decorator

# Constants for configuration settings
MAX_ITER = 1500
ALPHA = 0.02
L1_RATIO = 0.6
MINIMAL_SIMILARITY_THRESHOLD = 0.015
TOP_TWEETS_LIMIT = 15
pd.set_option("display.max_colwidth", None)


@section_printing_decorator("TOPIC MODELING")
def show_topics_modeled_with_nmf(
    dataset: DataSet, paths_config: PathsConfig, num_topics: int = 4,
) -> pd.DataFrame:

    nmf = NMF(
        n_components=num_topics, alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=MAX_ITER
    ).fit(dataset.tfidf_for_aggregated_tweets)

    tfidf_feature_names = dataset.tfidf_vectorizer.get_feature_names_out()
    KeyPhraseExtractor.print_top_words_for_all_topics(nmf, tfidf_feature_names, TOP_TWEETS_LIMIT)

    tweets_with_topic_assignment = assign_topics_to_tweets(
        dataset.tfidf,
        dataset.initial_tweets_array,
        dataset.original_tweets_array,
        dataset.initial_df[CREATED_AT_COLUMN_NAME].to_numpy(),
        nmf,
    )

    for topic_index in range(num_topics):
        print(f"\n=====TOP FOR TOPIC {topic_index}=====")
        topic_tweets = tweets_with_topic_assignment[tweets_with_topic_assignment["topic"] == topic_index]
        print(f'Tweet count for topic #{topic_index}: {len(topic_tweets)}')
        top_topic_tweets = topic_tweets.sort_values(by="topic_value", ascending=False)[:TOP_TWEETS_LIMIT]
        top_topic_tweets.to_csv(os.path.join(paths_config.results_dir, f"top_topic_{topic_index}.csv"))
        print("Top tweets associated with topic:")
        print(top_topic_tweets["tweet"])

    unassigned_tweets = tweets_with_topic_assignment[tweets_with_topic_assignment["topic"].isnull()]
    df_unassigned_sample = unassigned_tweets[:20]
    print(f"Tweet count not associated with any topic: {len(df_unassigned_sample)}")
    df_unassigned_sample.to_csv(os.path.join(paths_config.results_dir, "unassigned_topics_examples.csv"))

    return tweets_with_topic_assignment


def assign_topics_to_tweets(
    tfidf_matrix: np.ndarray,
    initial_tweets: List[str],
    original_tweets: List[str],
    created_at_column: np.ndarray,
    nmf_model
) -> pd.DataFrame:
    tfidf_topic_similarity = nmf_model.transform(tfidf_matrix)

    topic_assignments = np.apply_along_axis(
        lambda row: (max(row), int(np.argmax(row))) if max(row) > MINIMAL_SIMILARITY_THRESHOLD else (None, None),
        1,
        tfidf_topic_similarity,
    )

    tweets_with_topic_assignment = pd.DataFrame(
        topic_assignments, columns=["topic_value", "topic"]
    )
    tweets_with_topic_assignment.insert(loc=0, column="tweet", value=initial_tweets)
    tweets_with_topic_assignment.insert(loc=1, column="original_tweet", value=original_tweets)
    tweets_with_topic_assignment.insert(loc=1, column=CREATED_AT_COLUMN_NAME, value=created_at_column)

    return tweets_with_topic_assignment

