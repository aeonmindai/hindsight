# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/trees/sklearn_wrapper.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Numpy-only stand-in for sklearn's KDTree, BallTree, NearestNeighbors and KMeans.

Hindsight does not depend on scikit-learn (it pulls in scipy). The surprisal trees only need exact
Euclidean k-nearest-neighbour queries and a small k-means, so both are done here with numpy.
KDTree and BallTree return the same exact neighbours, so ``tree_type`` only names the variant.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import SurprisalTree


def knn_query(
    points: NDArray[np.floating[Any]], queries: NDArray[np.floating[Any]], k: int
) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
    """Exact k nearest neighbours of each query among ``points``: (distances, indices), nearest first."""
    diff = queries[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    k = min(k, points.shape[0])
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(dist, order, axis=1), order


def kmeans(
    points: NDArray[np.floating[Any]], n_clusters: int, random_state: int = 42, max_iter: int = 300
) -> NDArray[np.floating[Any]]:
    """Lloyd's k-means with k-means++ seeding. Returns the cluster centres."""
    rng = np.random.default_rng(random_state)
    n = points.shape[0]
    centres = np.empty((n_clusters, points.shape[1]), dtype=float)
    centres[0] = points[rng.integers(n)]
    closest = np.sum((points - centres[0]) ** 2, axis=1)
    for c in range(1, n_clusters):
        total = closest.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=closest / total))
        centres[c] = points[idx]
        closest = np.minimum(closest, np.sum((points - centres[c]) ** 2, axis=1))
    labels = np.full(n, -1)
    for _ in range(max_iter):
        dist = np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            members = points[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
    return centres


class SklearnTreeWrapper(SurprisalTree):
    """
    Wrapper for KDTree and BallTree style surprisal computation (exact kNN in numpy).
    Uses density estimation via k-nearest neighbors.
    """

    tree_type: str
    k: int
    points: list[NDArray[np.floating[Any]]]
    tree: NDArray[np.floating[Any]] | None
    total_points: int

    def __init__(self, tree_type: str = "kd", k: int = 5, max_leaf_size: int = 10) -> None:
        super().__init__(max_leaf_size)
        if tree_type not in ("kd", "ball"):
            raise ValueError(f"Unknown tree type: {tree_type}")
        self.tree_type = tree_type
        self.k = k
        self.points = []
        self.tree = None

    def insert(self, point: np.ndarray) -> None:
        self.points.append(point)
        self.total_points += 1
        self._rebuild_tree()

    def batch_insert(self, points: np.ndarray) -> None:
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        if len(self.points) == 0:
            return
        self.tree = np.array(self.points)

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal using k-NN density estimation.
        S(e) ≈ log(V_k(e)) where V_k is the volume of k-ball
        """
        if self.tree is None or len(self.points) < self.k:
            return float("inf")

        k_actual = min(self.k, len(self.points))
        distances, _indices = knn_query(self.tree, np.asarray([point], dtype=float), k_actual)

        avg_distance: float = float(np.mean(distances[0]))

        dim = point.shape[0]
        surprisal_value: float = dim * np.log(avg_distance + 1e-10)

        return surprisal_value
