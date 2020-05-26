from typing import List

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


def transform_matrix_with_count_vectorizer(df: List[str]) -> csr_matrix:
    count_vectorizer = CountVectorizer()
    count_vectorized_matrix = count_vectorizer.fit_transform(df)
    return count_vectorized_matrix
