# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_trees.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Surprisal trees: the factory, and the numpy-only stand-ins for scikit-learn (ported from Honcho's tests/dreamer/test_trees.py)."""

import random

import numpy as np
import pytest

from hindsight_api.people.dreamer.trees import (
    CoverTree,
    GraphSurprisal,
    LSHSurprisal,
    PrototypeSurprisal,
    RPTree,
    SklearnTreeWrapper,
    SurprisalTree,
    create_tree,
)

ALL_TREE_TYPES = [
    "kdtree",
    "balltree",
    "rptree",
    "covertree",
    "lsh",
    "graph",
    "prototype",
]

EXPECTED_CLASS = {
    "kdtree": SklearnTreeWrapper,
    "balltree": SklearnTreeWrapper,
    "rptree": RPTree,
    "covertree": CoverTree,
    "lsh": LSHSurprisal,
    "graph": GraphSurprisal,
    "prototype": PrototypeSurprisal,
}


@pytest.mark.parametrize("tree_type", ALL_TREE_TYPES)
def test_create_tree_accepts_uniform_k_kwarg(tree_type: str):
    tree = create_tree(tree_type=tree_type, k=5)
    assert isinstance(tree, SurprisalTree)
    assert isinstance(tree, EXPECTED_CLASS[tree_type])


@pytest.mark.parametrize("tree_type", ALL_TREE_TYPES)
def test_create_tree_without_k(tree_type: str):
    """The factory should also work when no ``k`` is supplied."""
    tree = create_tree(tree_type=tree_type)
    assert isinstance(tree, EXPECTED_CLASS[tree_type])


def test_create_tree_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown tree type"):
        create_tree(tree_type="not_a_tree")


@pytest.mark.parametrize("tree_type", ALL_TREE_TYPES)
def test_every_tree_scores_an_outlier_higher_than_a_cluster_member(tree_type: str):
    """Without scikit-learn every tree type still works, and a far point is more surprising than a near one for the neighbour-based trees."""
    # Some trees (lsh, rptree) draw random projections from the global generators.
    np.random.seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 0.05, size=(60, 8)).astype(np.float32)
    tree = create_tree(tree_type=tree_type, k=5)
    tree.batch_insert(cluster)

    near = tree.surprisal(cluster[0] + 0.01)
    far = tree.surprisal(np.full(8, 3.0, dtype=np.float32))

    assert np.isfinite(near) or np.isinf(near)
    assert not np.isnan(far)
    # The neighbour- and distance-based trees score a far point higher. The randomised trees (rptree,
    # covertree, lsh) are copied unchanged and only checked to work.
    if tree_type in ("kdtree", "balltree", "graph", "prototype"):
        assert far > near


def test_knn_query_matches_brute_force():
    from hindsight_api.people.dreamer.trees.sklearn_wrapper import knn_query

    rng = np.random.default_rng(1)
    points = rng.normal(size=(30, 4))
    query = rng.normal(size=(1, 4))
    distances, indices = knn_query(points, query, 3)
    expected = np.argsort(np.linalg.norm(points - query, axis=1))[:3]
    assert list(indices[0]) == list(expected)
    assert distances[0][0] == pytest.approx(np.linalg.norm(points[expected[0]] - query[0]))


def test_kmeans_finds_separated_clusters():
    from hindsight_api.people.dreamer.trees.sklearn_wrapper import kmeans

    rng = np.random.default_rng(2)
    points = np.vstack([rng.normal(0, 0.05, (20, 2)), rng.normal(5, 0.05, (20, 2))])
    centres = kmeans(points, 2)
    assert sorted(round(float(c)) for c in centres.mean(axis=1)) == [0, 5]
