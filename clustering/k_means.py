from collections import Counter

from scipy.sparse.csr import csr_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

NUMBER_OF_CLUSTERS = 3


def cluster_data_with_k_means(
    count_vectorized_matrix: csr_matrix, normalized_corpus
):  # todo delete the latter

    model = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        max_iter=1000,
        n_init=10,
        # verbose=2,
    )
    k_means_matrix = model.fit_predict(count_vectorized_matrix)

    trues = [i for i, x in enumerate(k_means_matrix == 0) if x]
    print(normalized_corpus[trues[0]])
    plt.scatter(
        count_vectorized_matrix[k_means_matrix == 0, 0],
        count_vectorized_matrix[k_means_matrix == 0, 1],
        s=5,
        c="red",
    )
    plt.scatter(
        count_vectorized_matrix[k_means_matrix == 1, 0],
        count_vectorized_matrix[k_means_matrix == 1, 1],
        s=5,
        c="green",
    )
    plt.scatter(
        count_vectorized_matrix[k_means_matrix == 2, 0],
        count_vectorized_matrix[k_means_matrix == 2, 1],
        s=5,
        c="blue",
    )

    plt.scatter(
        model.cluster_centers_[0, 0], model.cluster_centers_[0, 1], s=150, c="black"
    )
    plt.scatter(
        model.cluster_centers_[1, 0], model.cluster_centers_[1, 1], s=150, c="black"
    )
    plt.scatter(
        model.cluster_centers_[2, 0], model.cluster_centers_[2, 1], s=150, c="black"
    )
    # plt.scatter(
    #     model.cluster_centers_[:, 0],
    #     model.cluster_centers_[:, 1],
    #     s=300, c='black',
    #     label='Centroids'
    # )
    plt.show()
    return model
