from data.loaders import DataLoader
from data.paths import LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from corpus.corpus_transformation_pipeline import CorpusTransformer
from configs.config_loader import ConfigLoader
from information_extraction.keyphrase_extraction import get_top_ngrams


def run_liverpool_watford_processing():
    configs = ConfigLoader.load()
    df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    normalized_corpus = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets, hyper_parameters_config=configs.settings["setting_1"],
    )
    print(get_top_ngrams(normalized_corpus, ngram_length=1, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=2, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=3, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=4, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=5, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=6, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=7, ngrams_limit=10))
    print(get_top_ngrams(normalized_corpus, ngram_length=8, ngrams_limit=10))

    # print(get_tfidf_weighted_keyphrases(normalized_corpus))

    """
    SCRATCHES:
    from feature_extraction.word2vector import get_trained_word_2_vector_model
    from corpus.feature_extraction.tfidf import create_df_with_tfidf_feature_vectors
    from corpus.feature_extraction.document_similarity import get_cosine_similarity_df_from_tfidf_matrix
    
    word_2_vector_model = get_trained_word_2_vector_model(normalized_corpus)

    ++++ CLUSTERING ++++
    vectorized_matrix = transform_matrix_with_count_vectorizer(normalized_corpus)

    k_means_model = cluster_data_with_k_means(word_2_vector_model.vectors, normalized_corpus)
    try:
        first_cluster_center = k_means_model.cluster_centers_[0]
        print(list(map(lambda word: word[0], word_2_vector_model.similar_by_vector(first_cluster_center, topn=20))))

        second_cluster_center = k_means_model.cluster_centers_[1]
        print(list(map(lambda word: word[0], word_2_vector_model.similar_by_vector(second_cluster_center, topn=20))))

        third_cluster_center = k_means_model.cluster_centers_[2]
        print(list(map(lambda word: word[0], word_2_vector_model.similar_by_vector(third_cluster_center, topn=20))))
    except:
        pass

    ++++ COSINUS SIMILARITY ++++
    tfidf_df = create_df_with_tfidf_feature_vectors(corpus=normalized_corpus)
    cosine_similairty_df = get_cosine_similarity_df_from_tfidf_matrix(tfidf_df)
    print(cosine_similairty_df)
    """


if __name__ == "__main__":
    run_liverpool_watford_processing()
