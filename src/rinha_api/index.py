from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import msgspec
import numpy as np


VECTORS_FILE = "vectors.f32.npy"
HALF_FILE = "vectors.f16.npy"
QUANTIZED_FILE = "vectors.u1.npy"
LABELS_FILE = "labels.u1.npy"
CENTROIDS_FILE = "centroids.f32.npy"
OFFSETS_FILE = "offsets.i8.npy"
TREE_FILE = "tree.npz"
META_FILE = "index.json"
INDEX_ALGORITHM = "ivf-flat-q8-rerank-f16"
DEFAULT_CELLS = 4096
DEFAULT_NPROBE = 1
DEFAULT_FAST_MARGIN = 1.0
DEFAULT_TREE_CONFIDENCE = 0.95
DEFAULT_RERANK_K = 16
TRUTHY = {"1", "true", "yes", "on"}


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in TRUTHY


class VectorIndex:
    def __init__(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
        tree: dict[str, np.ndarray] | None = None,
        rerank_vectors: np.ndarray | None = None,
        block_size: int = 65536,
        nprobe: int = DEFAULT_NPROBE,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != 14:
            raise ValueError("vectors must have shape (n, 14)")
        if labels.ndim != 1 or labels.shape[0] != vectors.shape[0]:
            raise ValueError("labels must have one entry per vector")
        if centroids is not None and (centroids.ndim != 2 or centroids.shape[1] != 14):
            raise ValueError("centroids must have shape (n, 14)")
        if offsets is not None:
            if centroids is None:
                raise ValueError("offsets require centroids")
            if offsets.ndim != 1 or offsets.shape[0] != centroids.shape[0] + 1:
                raise ValueError("offsets must have one more entry than centroids")
            if int(offsets[0]) != 0 or int(offsets[-1]) != labels.shape[0]:
                raise ValueError("offsets must span all vectors")
        if rerank_vectors is not None and rerank_vectors.shape != vectors.shape:
            raise ValueError("rerank vectors must have same shape as vectors")
        self.vectors = vectors.astype(np.uint8 if vectors.dtype == np.uint8 else np.float16, copy=False)
        self.rerank_vectors = (
            rerank_vectors.astype(np.float16, copy=False) if rerank_vectors is not None else None
        )
        self.labels = labels.astype(np.uint8, copy=False)
        self.centroids = centroids.astype(np.float32, copy=False) if centroids is not None else None
        self.centroid_norms = np.sum(self.centroids * self.centroids, axis=1) if self.centroids is not None else None
        self.offsets = offsets.astype(np.int64, copy=False) if offsets is not None else None
        self.tree = tree
        self.block_size = block_size
        self.nprobe = max(1, nprobe)
        self.fast_margin = float(os.getenv("RINHA_CELL_FAST_MARGIN", str(DEFAULT_FAST_MARGIN)))
        self.rerank_k = max(5, int(os.getenv("RINHA_RERANK_K", str(DEFAULT_RERANK_K))))
        self.tree_confidence = float(os.getenv("RINHA_TREE_CONFIDENCE", str(DEFAULT_TREE_CONFIDENCE)))
        self.cell_scores = self._build_cell_scores()
        self.vector_norms = self._build_vector_norms()

    def score(self, query: np.ndarray) -> float:
        if self.labels.shape[0] == 0:
            return 1.0
        tree_score = self._score_tree(query)
        if tree_score is not None:
            return tree_score
        if self.centroids is not None and self.offsets is not None:
            return self._score_ivf(query)
        return self._score_exact(query)

    def _score_exact(self, query: np.ndarray) -> float:
        neighbors = min(5, self.labels.shape[0])
        candidate_k = self._candidate_k(neighbors)
        use_quantized = self.vectors.dtype == np.uint8
        quantized = quantize_vector(query).astype(np.int16, copy=False) if use_quantized else None
        best_distances = np.full(
            candidate_k,
            np.iinfo(np.int32).max if use_quantized else np.inf,
            dtype=np.int32 if use_quantized else np.float32,
        )
        best_ids = np.full(candidate_k, -1, dtype=np.int64)

        for start in range(0, self.labels.shape[0], self.block_size):
            end = min(start + self.block_size, self.labels.shape[0])
            block = self.vectors[start:end]
            if use_quantized:
                block = block.astype(np.int16, copy=False)
                diff = block - quantized
                diff32 = diff.astype(np.int32, copy=False)
                distances = np.sum(diff32 * diff32, axis=1, dtype=np.int32)
            else:
                diff = block.astype(np.float32, copy=False) - query.astype(np.float32, copy=False)
                distances = np.sum(diff * diff, axis=1)
            take = min(candidate_k, distances.shape[0])
            local = np.argpartition(distances, take - 1)[:take]
            best_ids, best_distances = self._merge_top_k(
                best_ids[best_ids >= 0],
                best_distances[best_ids >= 0],
                start + local,
                distances[local],
                candidate_k,
            )

        best_ids = self._rerank_top_k(query, best_ids, neighbors)
        frauds = int(np.sum(self.labels[best_ids]))
        return frauds / float(neighbors)

    def _score_ivf(self, query: np.ndarray) -> float:
        assert self.centroids is not None
        assert self.offsets is not None
        assert self.centroid_norms is not None

        probes = min(self.nprobe, self.centroids.shape[0])
        query_f32 = query.astype(np.float32, copy=False)
        query_norm = float(np.sum(query_f32 * query_f32))
        center_distances = self.centroid_norms + query_norm - 2.0 * (self.centroids @ query_f32)
        if self.cell_scores is not None:
            nearest = int(np.argmin(center_distances))
            cell_score = float(self.cell_scores[nearest])
            if cell_score <= 0.5 - self.fast_margin or cell_score >= 0.5 + self.fast_margin:
                return cell_score
        cells = np.argpartition(center_distances, probes - 1)[:probes]
        use_quantized = self.vectors.dtype == np.uint8
        quantized = quantize_vector(query).astype(np.int16, copy=False) if use_quantized else None

        neighbors = min(5, self.labels.shape[0])
        candidate_k = self._candidate_k(neighbors)
        best_ids = np.array([], dtype=np.int64)
        best_distances = np.array([], dtype=np.int32 if use_quantized else np.float32)

        for cell in cells:
            start = int(self.offsets[cell])
            end = int(self.offsets[cell + 1])
            if end <= start:
                continue

            block = self.vectors[start:end]
            if use_quantized:
                block = block.astype(np.int16, copy=False)
                diff = block - quantized
                diff32 = diff.astype(np.int32, copy=False)
                distances = np.sum(diff32 * diff32, axis=1, dtype=np.int32)
            elif self.vector_norms is not None:
                block = block.astype(np.float32, copy=False)
                distances = self.vector_norms[start:end] + query_norm - 2.0 * (block @ query_f32)
            else:
                diff = block.astype(np.float32, copy=False) - query_f32
                distances = np.sum(diff * diff, axis=1)

            take = min(candidate_k, distances.shape[0])
            local = np.argpartition(distances, take - 1)[:take]
            best_ids, best_distances = self._merge_top_k(
                best_ids,
                best_distances,
                start + local,
                distances[local],
                candidate_k,
            )

        if best_ids.size == 0:
            return self._score_exact(query)

        best_ids = self._rerank_top_k(query_f32, best_ids, neighbors)
        frauds = int(np.sum(self.labels[best_ids]))
        return frauds / float(best_ids.shape[0])

    @staticmethod
    def _merge_top_k(
        current_ids: np.ndarray,
        current_distances: np.ndarray,
        candidate_ids: np.ndarray,
        candidate_distances: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if candidate_ids.size == 0:
            return current_ids, current_distances
        if current_ids.size == 0:
            take = min(k, candidate_ids.shape[0])
            keep = np.argpartition(candidate_distances, take - 1)[:take]
            return candidate_ids[keep], candidate_distances[keep]

        merged_ids = np.concatenate((current_ids, candidate_ids))
        merged_distances = np.concatenate((current_distances, candidate_distances))
        take = min(k, merged_ids.shape[0])
        keep = np.argpartition(merged_distances, take - 1)[:take]
        return merged_ids[keep], merged_distances[keep]

    def _candidate_k(self, neighbors: int) -> int:
        if self.vectors.dtype == np.uint8 and self.rerank_vectors is not None:
            return min(self.rerank_k, self.labels.shape[0])
        return neighbors

    def _rerank_top_k(self, query: np.ndarray, candidate_ids: np.ndarray, k: int) -> np.ndarray:
        if self.rerank_vectors is None or candidate_ids.shape[0] <= k:
            return candidate_ids[:k]

        vectors = self.rerank_vectors[candidate_ids].astype(np.float32, copy=False)
        query_f32 = query.astype(np.float32, copy=False)
        diff = vectors - query_f32
        distances = np.sum(diff * diff, axis=1, dtype=np.float32)
        take = min(k, candidate_ids.shape[0])
        best = np.argpartition(distances, take - 1)[:take]
        return candidate_ids[best]

    def _build_cell_scores(self) -> np.ndarray | None:
        if self.centroids is None or self.offsets is None:
            return None

        scores = np.full(self.centroids.shape[0], 0.5, dtype=np.float32)
        for cell in range(self.centroids.shape[0]):
            start = int(self.offsets[cell])
            end = int(self.offsets[cell + 1])
            if start != end:
                scores[cell] = float(np.mean(self.labels[start:end]))
        return scores

    def _build_vector_norms(self) -> np.ndarray | None:
        if self.vectors.dtype == np.uint8:
            return None

        norms = np.empty(self.vectors.shape[0], dtype=np.float32)
        for start in range(0, self.vectors.shape[0], self.block_size):
            end = min(start + self.block_size, self.vectors.shape[0])
            block = self.vectors[start:end].astype(np.float32, copy=False)
            norms[start:end] = np.sum(block * block, axis=1)
        return norms

    def _score_tree(self, query: np.ndarray) -> float | None:
        if self.tree is None:
            return None

        features = self.tree["features"]
        thresholds = self.tree["thresholds"]
        scores = self.tree["scores"]
        left = self.tree["left"]
        right = self.tree["right"]

        node = 0
        while features[node] >= 0:
            feature = int(features[node])
            node = int(left[node]) if query[feature] <= thresholds[node] else int(right[node])

        score = float(scores[node])
        if score <= 1.0 - self.tree_confidence or score >= self.tree_confidence:
            return score
        return None

    def warmup(self) -> None:
        if self.centroids is not None:
            _ = float(np.sum(self.centroids, dtype=np.float64))
        if self.offsets is not None:
            _ = int(np.sum(self.offsets, dtype=np.int64))
        if self.labels.shape[0] > 0:
            _ = int(np.sum(self.labels, dtype=np.uint64))
        if self.vectors.shape[0] > 0:
            if self.vectors.dtype == np.uint8:
                _ = int(np.sum(self.vectors, dtype=np.uint64))
            else:
                _ = float(np.sum(self.vectors, dtype=np.float64))
        if self.rerank_vectors is not None and self.rerank_vectors.shape[0] > 0:
            _ = float(np.sum(self.rerank_vectors, dtype=np.float64))
        if self.tree is not None:
            for values in self.tree.values():
                _ = values.shape


def load_index(index_dir: Path) -> VectorIndex:
    mmap_mode = None if env_flag("RINHA_INDEX_PRELOAD") else "r"
    rerank_mmap_mode = None if env_flag("RINHA_RERANK_PRELOAD") else "r"
    rerank_vectors = None
    half_path = index_dir / HALF_FILE
    if (index_dir / QUANTIZED_FILE).exists():
        vectors = np.load(index_dir / QUANTIZED_FILE, mmap_mode=mmap_mode)
        if half_path.exists():
            rerank_vectors = np.load(half_path, mmap_mode=rerank_mmap_mode)
    elif half_path.exists():
        vectors = np.load(half_path, mmap_mode=mmap_mode)
    else:
        vectors = quantize_vectors(np.load(index_dir / VECTORS_FILE, mmap_mode=mmap_mode))
    labels = np.load(index_dir / LABELS_FILE, mmap_mode=mmap_mode)
    centroids = None
    offsets = None
    if (index_dir / CENTROIDS_FILE).exists() and (index_dir / OFFSETS_FILE).exists():
        centroids = np.load(index_dir / CENTROIDS_FILE, mmap_mode=mmap_mode)
        offsets = np.load(index_dir / OFFSETS_FILE, mmap_mode=mmap_mode)
    tree = None
    if (index_dir / TREE_FILE).exists():
        tree_file = np.load(index_dir / TREE_FILE)
        tree = {key: tree_file[key] for key in tree_file.files}
    nprobe = int(os.getenv("RINHA_IVF_NPROBE", str(DEFAULT_NPROBE)))
    return VectorIndex(
        vectors,
        labels,
        centroids=centroids,
        offsets=offsets,
        tree=tree,
        rerank_vectors=rerank_vectors,
        nprobe=nprobe,
    )


def build_index(references_path: Path, index_dir: Path, cells: int | None = None) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    decoder = msgspec.json.Decoder(type=list[dict[str, Any]])

    if references_path.suffix == ".gz":
        with gzip.open(references_path, "rb") as file:
            records = decoder.decode(file.read())
    else:
        records = decoder.decode(references_path.read_bytes())

    vectors = np.empty((len(records), 14), dtype=np.float32)
    labels = np.empty(len(records), dtype=np.uint8)

    for index, record in enumerate(records):
        vectors[index] = record["vector"]
        labels[index] = 1 if record["label"] == "fraud" else 0

    quantized = quantize_vectors(vectors)
    cell_count = min(max(1, cells or int(os.getenv("RINHA_IVF_CELLS", str(DEFAULT_CELLS)))), len(records))
    centroids = train_centroids(vectors, cell_count)
    assignments = assign_centroids(vectors, centroids)
    counts = np.bincount(assignments, minlength=cell_count).astype(np.int64)
    offsets = np.empty(cell_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(assignments, kind="stable")

    np.save(index_dir / QUANTIZED_FILE, quantized[order])
    np.save(index_dir / HALF_FILE, vectors[order].astype(np.float16, copy=False))
    np.save(index_dir / LABELS_FILE, labels[order])
    np.save(index_dir / CENTROIDS_FILE, centroids)
    np.save(index_dir / OFFSETS_FILE, offsets)
    tree = train_confident_tree(vectors, labels)
    np.savez_compressed(index_dir / TREE_FILE, **tree)

    stat = references_path.stat()
    meta = {
        "algorithm": INDEX_ALGORITHM,
        "dimensions": 14,
        "count": len(records),
        "cells": cell_count,
        "ivf_sample": int(os.getenv("RINHA_IVF_SAMPLE", "50000")),
        "ivf_iterations": int(os.getenv("RINHA_IVF_ITERATIONS", "4")),
        "tree_sample": int(os.getenv("RINHA_TREE_SAMPLE", "500000")),
        "tree_depth": int(os.getenv("RINHA_TREE_DEPTH", "10")),
        "tree_quantiles": int(os.getenv("RINHA_TREE_QUANTILES", "199")),
        "tree_min_leaf": int(os.getenv("RINHA_TREE_MIN_LEAF", "50")),
        "source_name": references_path.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")


def empty_index() -> VectorIndex:
    return VectorIndex(np.empty((0, 14), dtype=np.uint8), np.empty(0, dtype=np.uint8))


def quantize_vectors(vectors: np.ndarray) -> np.ndarray:
    clipped = np.clip(vectors, -1.0, 1.0)
    return np.rint((clipped + 1.0) * 127.5).astype(np.uint8)


def quantize_vector(vector: np.ndarray) -> np.ndarray:
    clipped = np.clip(vector, -1.0, 1.0)
    return np.rint((clipped + 1.0) * 127.5).astype(np.uint8)


def train_centroids(vectors: np.ndarray, cells: int) -> np.ndarray:
    if cells == 1:
        return vectors.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)

    rng = np.random.default_rng(42)
    sample_size = min(vectors.shape[0], max(cells, int(os.getenv("RINHA_IVF_SAMPLE", "50000"))))
    sample_ids = rng.choice(vectors.shape[0], size=sample_size, replace=False)
    sample = vectors[sample_ids].astype(np.float32, copy=False)
    centroid_ids = rng.choice(sample.shape[0], size=cells, replace=False)
    centroids = sample[centroid_ids].copy()

    iterations = max(0, int(os.getenv("RINHA_IVF_ITERATIONS", "4")))
    for _ in range(iterations):
        assignments = assign_centroids(sample, centroids)
        sums = np.zeros((cells, vectors.shape[1]), dtype=np.float64)
        counts = np.bincount(assignments, minlength=cells)
        np.add.at(sums, assignments, sample)
        populated = counts > 0
        centroids[populated] = (sums[populated] / counts[populated, None]).astype(np.float32)

    return centroids


def assign_centroids(vectors: np.ndarray, centroids: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    assignments = np.empty(vectors.shape[0], dtype=np.int32)
    centroid_norms = np.sum(centroids * centroids, axis=1)

    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        batch = vectors[start:end].astype(np.float32, copy=False)
        distances = np.sum(batch * batch, axis=1)[:, None] + centroid_norms[None, :] - 2.0 * (batch @ centroids.T)
        assignments[start:end] = np.argmin(distances, axis=1)

    return assignments


def train_confident_tree(vectors: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    sample_size = min(vectors.shape[0], int(os.getenv("RINHA_TREE_SAMPLE", "500000")))
    sample_ids = rng.choice(vectors.shape[0], size=sample_size, replace=False)
    sample_vectors = vectors[sample_ids]
    sample_labels = labels[sample_ids]
    max_depth = int(os.getenv("RINHA_TREE_DEPTH", "10"))
    min_leaf = int(os.getenv("RINHA_TREE_MIN_LEAF", "50"))
    quantiles = int(os.getenv("RINHA_TREE_QUANTILES", "199"))
    candidate_thresholds = [
        np.unique(
            np.quantile(sample_vectors[:, feature], np.linspace(0.01, 0.99, quantiles)).astype(np.float32)
        )
        for feature in range(sample_vectors.shape[1])
    ]

    nodes: list[list[float | int]] = []

    def best_split(ids: np.ndarray) -> tuple[int | None, float, np.ndarray | None, np.ndarray | None]:
        node_labels = sample_labels[ids]
        total = int(node_labels.sum())
        count = ids.shape[0]
        base = total * (count - total) / count
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold = 0.0
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None

        for feature, thresholds in enumerate(candidate_thresholds):
            values = sample_vectors[ids, feature]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_labels = node_labels[order]
            cumulative = np.cumsum(sorted_labels)
            positions = np.searchsorted(sorted_values, thresholds, side="right")

            for position, threshold in zip(positions, thresholds):
                right_count = count - position
                if position < min_leaf or right_count < min_leaf:
                    continue

                left_positive = int(cumulative[position - 1])
                right_positive = total - left_positive
                left_impurity = left_positive * (position - left_positive) / position
                right_impurity = right_positive * (right_count - right_positive) / right_count
                gain = base - left_impurity - right_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_left = ids[order[:position]]
                    best_right = ids[order[position:]]

        return best_feature, best_threshold, best_left, best_right

    def build(ids: np.ndarray, depth: int) -> int:
        node_id = len(nodes)
        score = float(sample_labels[ids].mean()) if ids.shape[0] else 0.5
        nodes.append([score, -1, 0.0, -1, -1])

        if depth >= max_depth or ids.shape[0] < min_leaf * 2 or score == 0.0 or score == 1.0:
            return node_id

        feature, threshold, left_ids, right_ids = best_split(ids)
        if feature is None or left_ids is None or right_ids is None:
            return node_id

        left_node = build(left_ids, depth + 1)
        right_node = build(right_ids, depth + 1)
        nodes[node_id] = [score, feature, threshold, left_node, right_node]
        return node_id

    build(np.arange(sample_vectors.shape[0], dtype=np.int32), 0)
    raw = np.asarray(nodes, dtype=np.float32)
    return {
        "scores": raw[:, 0].astype(np.float32),
        "features": raw[:, 1].astype(np.int16),
        "thresholds": raw[:, 2].astype(np.float32),
        "left": raw[:, 3].astype(np.int32),
        "right": raw[:, 4].astype(np.int32),
    }


def index_matches_source(index_dir: Path, references_path: Path) -> bool:
    required = [QUANTIZED_FILE, HALF_FILE, LABELS_FILE, CENTROIDS_FILE, OFFSETS_FILE, TREE_FILE, META_FILE]
    if any(not (index_dir / filename).exists() for filename in required):
        return False

    try:
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        stat = references_path.stat()
    except (OSError, ValueError):
        return False

    return (
        meta.get("algorithm") == INDEX_ALGORITHM
        and meta.get("cells") == int(os.getenv("RINHA_IVF_CELLS", str(DEFAULT_CELLS)))
        and meta.get("ivf_sample") == int(os.getenv("RINHA_IVF_SAMPLE", "50000"))
        and meta.get("ivf_iterations") == int(os.getenv("RINHA_IVF_ITERATIONS", "4"))
        and meta.get("tree_sample") == int(os.getenv("RINHA_TREE_SAMPLE", "500000"))
        and meta.get("tree_depth") == int(os.getenv("RINHA_TREE_DEPTH", "10"))
        and meta.get("tree_quantiles") == int(os.getenv("RINHA_TREE_QUANTILES", "199"))
        and meta.get("tree_min_leaf") == int(os.getenv("RINHA_TREE_MIN_LEAF", "50"))
        and meta.get("dimensions") == 14
        and meta.get("source_name") == references_path.name
        and meta.get("source_size") == stat.st_size
        and meta.get("source_mtime_ns") == stat.st_mtime_ns
    )
