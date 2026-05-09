import json

import numpy as np

from rinha_api.index import (
    CENTROIDS_FILE,
    HALF_FILE,
    LABELS_FILE,
    META_FILE,
    OFFSETS_FILE,
    QUANTIZED_FILE,
    TREE_FILE,
    VectorIndex,
    index_matches_source,
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


def test_index_matches_source_uses_metadata(tmp_path) -> None:
    references = tmp_path / "references.json"
    references.write_text("[]", encoding="utf-8")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    for filename in [HALF_FILE, LABELS_FILE, CENTROIDS_FILE, OFFSETS_FILE, TREE_FILE]:
        (index_dir / filename).write_bytes(b"placeholder")

    stat = references.stat()
    (index_dir / META_FILE).write_text(
        json.dumps(
            {
                "algorithm": "ivf-flat-f16",
                "cells": 4096,
                "dimensions": 14,
                "source_name": references.name,
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }
        ),
        encoding="utf-8",
    )

    assert index_matches_source(index_dir, references)

    references.write_text("[1]", encoding="utf-8")

    assert not index_matches_source(index_dir, references)
