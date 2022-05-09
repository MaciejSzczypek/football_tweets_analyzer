import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from data.column_names import CREATED_AT_COLUMN_NAME
from data.utils import DataSet
from information_extraction.keyphrase_extraction import print_top_words_for_all_topics
from utils.printing import section_printing_decorator


@section_printing_decorator
def show_topics_modeled_with_nmf(dataset: DataSet) -> pd.DataFrame:
    pd.set_option("display.max_colwidth", -1)
    print("5. TOPIC MODELING")
    print()
    max_iter = 1500
    n_of_topics = 4
    alpha = 0.02
    l1_ratio = 0.6
    print(f"NMF: alpha: {alpha}, l1_ratio: {l1_ratio}, max_iter: {max_iter}")
    nmf = NMF(
        n_components=n_of_topics, alpha=alpha, l1_ratio=l1_ratio,  # max_iter=max_iter
    ).fit(dataset.tfidf_for_aggregated_tweets)
    tfidf_feature_names = dataset.tfidf_vectorizer.get_feature_names()
    print_top_words_for_all_topics(nmf, tfidf_feature_names, 15)
    tfidf_topic_similarity = nmf.transform(dataset.tfidf)
    minimal_similarity_threshold = 0.015  # domyslnie 0.015
    tweets_with_topic_assignment = pd.DataFrame(
        np.apply_along_axis(
            lambda row: (max(row), int(np.argmax(row)))
            if max(row) > minimal_similarity_threshold
            else (None, None),
            1,
            tfidf_topic_similarity,
        ),
        columns=["topic_value", "topic"],
    )
    tweets_with_topic_assignment.insert(loc=0, column="tweet", value=dataset.initial_tweets_array)
    tweets_with_topic_assignment.insert(loc=1, column="original_tweet", value=dataset.original_tweets_array)
    tweets_with_topic_assignment.insert(
        loc=1, column=CREATED_AT_COLUMN_NAME, value=dataset.initial_df[CREATED_AT_COLUMN_NAME].to_numpy()
    )
    print(
        len(
            tweets_with_topic_assignment[
                (tweets_with_topic_assignment["topic"] == 0)
                | (tweets_with_topic_assignment["topic"] == 1)
                | (tweets_with_topic_assignment["topic"] == 2)
                | (tweets_with_topic_assignment["topic"] == 3)
            ]
        )
    )
    # top tweets per topic
    for topic_index in range(n_of_topics):
        print()
        print(f"=====TOP FOR TOPIC {topic_index}=====")
        topic_tweets = tweets_with_topic_assignment[
            tweets_with_topic_assignment["topic"] == topic_index
        ]
        print(topic_index, "====>", len(topic_tweets))
        top_topic_tweets = topic_tweets.sort_values(by="topic_value", ascending=False)[:15]
        top_topic_tweets.to_csv(f"results/top_topic_{topic_index}.csv")
        print(top_topic_tweets)

    print(
        "@@@@@",
        len(tweets_with_topic_assignment[
            (tweets_with_topic_assignment["topic"] != 0)
            & (tweets_with_topic_assignment["topic"] != 1)
            & (tweets_with_topic_assignment["topic"] != 2)
            & (tweets_with_topic_assignment["topic"] != 3)
        ])
    )
    df_unassigned = tweets_with_topic_assignment[
            (tweets_with_topic_assignment["topic"] != 0)
            & (tweets_with_topic_assignment["topic"] != 1)
            & (tweets_with_topic_assignment["topic"] != 2)
            & (tweets_with_topic_assignment["topic"] != 3)
    ][:20]
    print(df_unassigned)
    df_unassigned.to_csv("results/unassigned_topics_examples.csv")

    return tweets_with_topic_assignment

