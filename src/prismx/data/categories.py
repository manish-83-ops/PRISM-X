"""PRISMX Derived Metadata Categories via Seeded MiniBatchKMeans and c-TF-IDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CLUSTER_DIR = REPO_ROOT / "data" / "manifests"

class CategoryManager:
    def __init__(self, n_clusters: int = 15, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans: MiniBatchKMeans | None = None
        self.cluster_labels: dict[int, str] = {}
        self.cluster_sizes: dict[int, int] = {}

    def fit(self, embeddings: np.ndarray, texts: list[str]) -> list[str]:
        """Fits MiniBatchKMeans and derives c-TF-IDF category slugs for each cluster."""
        print(f"Clustering {len(embeddings)} passages into {self.n_clusters} clusters (seed={self.seed})...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            batch_size=1024,
            n_init="auto",
        )
        cluster_assignments = self.kmeans.fit_predict(embeddings)

        # Count cluster sizes
        unique, counts = np.unique(cluster_assignments, return_counts=True)
        self.cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}

        # Aggregate texts per cluster for c-TF-IDF
        cluster_docs = [""] * self.n_clusters
        for text, c_id in zip(texts, cluster_assignments):
            cluster_docs[c_id] += " " + text

        # Compute c-TF-IDF
        vectorizer = CountVectorizer(stop_words="english", max_features=10000)
        X = vectorizer.fit_transform(cluster_docs).toarray()
        words = vectorizer.get_feature_names_out()

        # TF per cluster normalized by total words in cluster
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        # IDF across clusters
        df = (X > 0).sum(axis=0)
        idf = np.log((self.n_clusters + 1) / (df + 1)) + 1
        c_tfidf = tf * idf

        # Extract top 2 terms per cluster for slug
        for c_id in range(self.n_clusters):
            top_indices = np.argsort(c_tfidf[c_id])[::-1][:2]
            top_words = [words[idx] for idx in top_indices]
            slug = "-".join(top_words).lower()
            self.cluster_labels[c_id] = slug

        return [self.cluster_labels[c_id] for c_id in cluster_assignments]

    def predict(self, embedding: np.ndarray) -> str:
        """Assigns an embedding to the nearest centroid and returns its category slug."""
        if self.kmeans is None:
            raise RuntimeError("CategoryManager must be fitted or loaded before prediction.")
        emb = embedding.reshape(1, -1)
        c_id = int(self.kmeans.predict(emb)[0])
        return self.cluster_labels.get(c_id, "general")

    def save(self, target_dir: Path | None = None) -> None:
        target_dir = target_dir or DEFAULT_CLUSTER_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save centroids
        centroids_path = target_dir / "cluster_centroids.npy"
        np.save(centroids_path, self.kmeans.cluster_centers_)

        # Save metadata
        meta = {
            "n_clusters": self.n_clusters,
            "seed": self.seed,
            "cluster_labels": {str(k): v for k, v in self.cluster_labels.items()},
            "cluster_sizes": {str(k): v for k, v in self.cluster_sizes.items()},
        }
        with open(target_dir / "cluster_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, target_dir: Path | None = None) -> None:
        target_dir = target_dir or DEFAULT_CLUSTER_DIR
        meta_path = target_dir / "cluster_metadata.json"
        centroids_path = target_dir / "cluster_centroids.npy"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.n_clusters = meta["n_clusters"]
        self.seed = meta["seed"]
        self.cluster_labels = {int(k): v for k, v in meta["cluster_labels"].items()}
        self.cluster_sizes = {int(k): v for k, v in meta["cluster_sizes"].items()}

        centroids = np.load(centroids_path)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.seed)
        self.kmeans.cluster_centers_ = centroids
