import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_cosine_similarity_df_from_tfidf_matrix(tfidf_df: pd.DataFrame) -> pd.DataFrame:
    cosine_similarity_matrix = cosine_similarity(
        tfidf_df
    )
    cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix)
    return cosine_similarity_df
