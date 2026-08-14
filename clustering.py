"""
Unsupervised -- Job / Role Clustering.

Reduces the job TF-IDF matrix with TruncatedSVD (LSA), runs K-Means over a
range of k, picks k via elbow (inertia) + silhouette score, then fits the
final model and saves cluster assignments + top terms per cluster.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from src.config import (
    JOB_CORPUS_PARQUET,
    JOB_TFIDF_VECTORIZER_PATH,
    JOB_TFIDF_MATRIX_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
)

SVD_COMPONENTS = 100
K_RANGE = range(4, 21, 2)  # test k = 4,6,...,20
SVD_PATH = MODELS_DIR / "job_svd.pkl"
KMEANS_PATH = MODELS_DIR / "job_kmeans.pkl"
CLUSTER_ASSIGNMENTS_PATH = MODELS_DIR / "job_clusters.parquet"


def reduce_dimensions(matrix, n_components: int = SVD_COMPONENTS):
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(matrix)
    print(f"SVD: {matrix.shape} -> {reduced.shape}, "
          f"explained variance={svd.explained_variance_ratio_.sum():.3f}")
    return svd, reduced


def select_k(reduced, k_range=K_RANGE, sample_size: int = 8000):
    """Elbow (inertia) + silhouette score over a range of k. Silhouette is
    computed on a random subsample since it's O(n^2) and the corpus is ~70k rows."""
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(reduced), size=min(sample_size, len(reduced)), replace=False)

    inertias, silhouettes = [], []
    for k in k_range:
        # MiniBatchKMeans for the k-search: full KMeans over 9 k-values x 70k
        # rows x n_init=10 is too slow for interactive iteration; MiniBatch
        # gives a fast, close-enough proxy for choosing k. The final model
        # below is refit with full KMeans for quality.
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, batch_size=2000)
        labels = km.fit_predict(reduced)
        inertias.append(km.inertia_)
        sil = silhouette_score(reduced[sample_idx], labels[sample_idx])
        silhouettes.append(sil)
        print(f"k={k:2d} | inertia={km.inertia_:10.1f} | silhouette={sil:.4f}")

    return list(k_range), inertias, silhouettes


def plot_k_selection(k_values, inertias, silhouettes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_values, inertias, marker="o", color="#2563eb")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_values, silhouettes, marker="o", color="#f97316")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    plt.tight_layout()
    out_path = FIGURES_DIR / "clustering_k_selection.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def top_terms_per_cluster(vectorizer, svd, kmeans, top_n: int = 12):
    """Project cluster centroids back to TF-IDF space to find representative terms."""
    centroids_tfidf_space = svd.inverse_transform(kmeans.cluster_centers_)
    terms = np.array(vectorizer.get_feature_names_out())
    result = {}
    for i, centroid in enumerate(centroids_tfidf_space):
        top_idx = np.argsort(centroid)[::-1][:top_n]
        result[i] = terms[top_idx].tolist()
    return result


def plot_cluster_scatter(reduced, labels, n_points: int = 5000):
    """2D PCA-of-SVD scatter colored by cluster, for the report/notebook."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(reduced), size=min(n_points, len(reduced)), replace=False)

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca_2d.fit_transform(reduced[idx])

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels[idx], cmap="tab20", s=6, alpha=0.6)
    ax.set_title("Job Clusters (PCA projection of SVD space)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    out_path = FIGURES_DIR / "clustering_pca_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def run_clustering(final_k: int = None):
    vectorizer = joblib.load(JOB_TFIDF_VECTORIZER_PATH)
    matrix = joblib.load(JOB_TFIDF_MATRIX_PATH)
    corpus = pd.read_parquet(JOB_CORPUS_PARQUET)

    svd, reduced = reduce_dimensions(matrix)

    if final_k is None:
        k_values, inertias, silhouettes = select_k(reduced)
        plot_k_selection(k_values, inertias, silhouettes)
        final_k = k_values[int(np.argmax(silhouettes))]
        print(f"\nSelected k={final_k} (best silhouette)")

    # Final model: full KMeans (better quality than MiniBatch) now that k is fixed.
    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10, max_iter=200)
    labels = kmeans.fit_predict(reduced)

    corpus["cluster"] = labels
    corpus[["job_id", "title", "cluster"]].to_parquet(CLUSTER_ASSIGNMENTS_PATH, index=False)

    joblib.dump(svd, SVD_PATH)
    joblib.dump(kmeans, KMEANS_PATH)

    terms = top_terms_per_cluster(vectorizer, svd, kmeans)
    print("\nTop terms per cluster:")
    for cid, words in terms.items():
        size = (labels == cid).sum()
        print(f"  Cluster {cid:2d} (n={size:5d}): {', '.join(words[:8])}")

    plot_cluster_scatter(reduced, labels)

    return kmeans, svd, corpus, terms


if __name__ == "__main__":
    run_clustering()
