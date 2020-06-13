from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def create_df_with_tfidf_feature_vectors(
    corpus: List[str], maximum_number_of_features: int = 100,
) -> pd.DataFrame:
    tfidf_vectorizer = TfidfVectorizer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        max_features=maximum_number_of_features,
        ngram_range=(2, 5),
        stop_words=stopwords.words('english'),
    )
    tfidf_vectorized_corpus = tfidf_vectorizer.fit_transform(corpus)
    corpus_vocabulary = tfidf_vectorizer.get_feature_names()
    df = pd.DataFrame(tfidf_vectorized_corpus.toarray(), columns=corpus_vocabulary,)
    return df
