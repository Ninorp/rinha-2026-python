import json

import numpy as np

from rinha_api.index import (
    BOUNDS_FILE,
    CENTROIDS_FILE,
    HALF_FILE,
    INDEX_ALGORITHM,
    LABELS_FILE,
    META_FILE,
    OFFSETS_FILE,
    QUANTIZED_FILE,
    TREE_FILE,
    VectorIndex,
    build_quantized_cell_bounds,
    index_matches_source,
    load_index,
    quantize_vectors,
)


def test_ivf_score_only_searches_nearby_cells() -> None:
    vectors = quantize_vectors(
        np.array(
            [
                [0.0] * 14,
                [0.02] * 14,
                [0.9] * 14,
                [0.95] * 14,
            ],
            dtype=np.float32,
        )
    )
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    centroids = np.array([[0.01] * 14, [0.925] * 14], dtype=np.float32)
    offsets = np.array([0, 2, 4], dtype=np.int64)
    index = VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, nprobe=1)

    assert index.score(np.array([0.93] * 14, dtype=np.float32)) == 1.0


def test_pure_cell_score_uses_cell_majority_fast_path() -> None:
    vectors = quantize_vectors(
        np.array(
            [
                [0.0] * 14,
                [0.02] * 14,
                [0.9] * 14,
                [0.95] * 14,
            ],
            dtype=np.float32,
        )
    )
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    centroids = np.array([[0.01] * 14, [0.925] * 14], dtype=np.float32)
    offsets = np.array([0, 2, 4], dtype=np.int64)
    index = VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, nprobe=1)

    assert index.score(np.array([0.01] * 14, dtype=np.float32)) == 0.0


def test_cell_prune_skips_distant_cells_without_changing_score() -> None:
    raw_vectors = np.array(
        [
            [0.00] * 14,
            [0.01] * 14,
            [0.02] * 14,
            [0.03] * 14,
            [0.04] * 14,
            [0.90] * 14,
            [0.91] * 14,
            [0.92] * 14,
            [0.93] * 14,
            [0.94] * 14,
        ],
        dtype=np.float32,
    )
    vectors = quantize_vectors(raw_vectors)
    labels = np.array([0] * 5 + [1] * 5, dtype=np.uint8)
    centroids = np.array([[0.02] * 14, [0.92] * 14], dtype=np.float32)
    offsets = np.array([0, 5, 10], dtype=np.int64)
    bounds = build_quantized_cell_bounds(vectors, centroids, offsets)
    query = np.array([0.01] * 14, dtype=np.float32)
    index = VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, bounds=bounds, nprobe=2)

    baseline = index.score(query)
    index.cell_prune = True
    index.probe_counts = []

    assert index.score(query) == baseline
    assert index.probe_counts == [1]


def test_cell_prune_keeps_cells_when_lower_bound_ties_worst_candidate() -> None:
    raw_vectors = np.array(
        [
            [0.00] * 14,
            [0.00] * 14,
            [0.00] * 14,
            [0.00] * 14,
            [0.00] * 14,
            [0.00] * 14,
        ],
        dtype=np.float32,
    )
    vectors = quantize_vectors(raw_vectors)
    labels = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    centroids = np.array([[0.00] * 14, [0.00] * 14], dtype=np.float32)
    offsets = np.array([0, 5, 6], dtype=np.int64)
    bounds = build_quantized_cell_bounds(vectors, centroids, offsets)
    index = VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, bounds=bounds, nprobe=2)
    index.cell_prune = True
    index.probe_counts = []

    index.score(np.array([0.00] * 14, dtype=np.float32))

    assert index.probe_counts == [2]


def test_batch_cell_scan_preserves_cell_prune_score() -> None:
    raw_vectors = np.array(
        [
            [0.00] * 14,
            [0.01] * 14,
            [0.02] * 14,
            [0.03] * 14,
            [0.04] * 14,
            [0.10] * 14,
            [0.11] * 14,
            [0.12] * 14,
            [0.13] * 14,
            [0.14] * 14,
            [0.90] * 14,
            [0.91] * 14,
            [0.92] * 14,
            [0.93] * 14,
            [0.94] * 14,
        ],
        dtype=np.float32,
    )
    vectors = quantize_vectors(raw_vectors)
    labels = np.array([0] * 5 + [1] * 5 + [1] * 5, dtype=np.uint8)
    centroids = np.array([[0.02] * 14, [0.12] * 14, [0.92] * 14], dtype=np.float32)
    offsets = np.array([0, 5, 10, 15], dtype=np.int64)
    bounds = build_quantized_cell_bounds(vectors, centroids, offsets)
    query = np.array([0.05] * 14, dtype=np.float32)
    index = VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, bounds=bounds, nprobe=3)
    index.cell_prune = True

    baseline = index.score(query)
    index.batch_cells = True
    index.probe_counts = []

    assert index.score(query) == baseline
    assert index.probe_counts == [2]


def test_deep_fallback_rechecks_boundary_ivf_score() -> None:
    raw_vectors = np.array(
        [
            [0.070] * 14,
            [0.071] * 14,
            [0.072] * 14,
            [0.080] * 14,
            [0.090] * 14,
            [0.050] * 14,
            [0.051] * 14,
            [0.052] * 14,
            [0.053] * 14,
            [0.054] * 14,
        ],
        dtype=np.float32,
    )
    vectors = quantize_vectors(raw_vectors)
    labels = np.array([1, 1, 1, 0, 0] + [0, 0, 0, 0, 0], dtype=np.uint8)
    centroids = np.array([[0.060] * 14, [0.200] * 14], dtype=np.float32)
    offsets = np.array([0, 5, 10], dtype=np.int64)
    bounds = build_quantized_cell_bounds(vectors, centroids, offsets)
    query = np.array([0.050] * 14, dtype=np.float32)
    index = VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, bounds=bounds, nprobe=1)
    index.cell_prune = True
    index.batch_cells = True

    assert index.score(query) == 0.6

    index.deep_nprobe = 2
    index.deep_score_counts = {3}

    assert index.score(query) < 0.6


def test_weighted_rerank_uses_neighbor_distance(monkeypatch) -> None:
    monkeypatch.setenv("RINHA_WEIGHTED_KNN", "1")
    monkeypatch.setenv("RINHA_WEIGHTED_EPS", "0.10")

    raw_vectors = np.array(
        [
            [0.00] * 14,
            [0.70] * 14,
            [0.71] * 14,
            [0.72] * 14,
            [0.73] * 14,
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 1, 1, 1], dtype=np.uint8)
    index = VectorIndex(quantize_vectors(raw_vectors), labels, rerank_vectors=raw_vectors)
    candidates = np.arange(5, dtype=np.int64)
    query = np.array([0.0] * 14, dtype=np.float32)

    assert index._score_reranked(query, candidates, 5) < 0.6


def test_load_index_prefers_quantized_vectors_and_keeps_half_precision_for_rerank(tmp_path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    vectors_f32 = np.array([[0.0] * 14, [1.0] * 14], dtype=np.float32)
    np.save(index_dir / QUANTIZED_FILE, quantize_vectors(vectors_f32))
    np.save(index_dir / HALF_FILE, vectors_f32.astype(np.float16))
    np.save(index_dir / LABELS_FILE, np.array([0, 1], dtype=np.uint8))

    index = load_index(index_dir)

    assert index.vectors.dtype == np.uint8
    assert index.rerank_vectors is not None
    assert index.rerank_vectors.dtype == np.float16


def test_confident_tree_score_short_circuits_index() -> None:
    vectors = quantize_vectors(np.array([[0.0] * 14, [1.0] * 14], dtype=np.float32))
    labels = np.array([0, 1], dtype=np.uint8)
    tree = {
        "scores": np.array([0.5, 0.0, 1.0], dtype=np.float32),
        "features": np.array([0, -1, -1], dtype=np.int16),
        "thresholds": np.array([0.5, 0.0, 0.0], dtype=np.float32),
        "left": np.array([1, -1, -1], dtype=np.int32),
        "right": np.array([2, -1, -1], dtype=np.int32),
    }
    index = VectorIndex(vectors, labels, tree=tree)

    assert index.score(np.array([0.9] * 14, dtype=np.float32)) == 1.0


def test_index_matches_source_uses_metadata(tmp_path, monkeypatch) -> None:
    references = tmp_path / "references.json"
    references.write_text("[]", encoding="utf-8")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    for filename in [QUANTIZED_FILE, HALF_FILE, LABELS_FILE, CENTROIDS_FILE, OFFSETS_FILE, BOUNDS_FILE, TREE_FILE]:
        (index_dir / filename).write_bytes(b"placeholder")

    stat = references.stat()
    (index_dir / META_FILE).write_text(
        json.dumps(
                {
                    "algorithm": INDEX_ALGORITHM,
                    "cells": 4096,
                    "ivf_sample": 50000,
                    "ivf_iterations": 4,
                    "tree_sample": 500000,
                    "tree_depth": 10,
                    "tree_quantiles": 199,
                    "tree_min_leaf": 50,
                    "dimensions": 14,
                    "source_name": references.name,
                    "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }
        ),
        encoding="utf-8",
    )

    assert index_matches_source(index_dir, references)

    monkeypatch.setenv("RINHA_TREE_DEPTH", "12")

    assert not index_matches_source(index_dir, references)

    monkeypatch.setenv("RINHA_TREE_DEPTH", "10")

    references.write_text("[1]", encoding="utf-8")

    assert not index_matches_source(index_dir, references)
