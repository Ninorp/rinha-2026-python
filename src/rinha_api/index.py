from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import msgspec
import numpy as np


VECTORS_FILE = "vectors.f32.npy"
QUANTIZED_FILE = "vectors.u1.npy"
LABELS_FILE = "labels.u1.npy"
CENTROIDS_FILE = "centroids.f32.npy"
OFFSETS_FILE = "offsets.i8.npy"
META_FILE = "index.json"
DEFAULT_CELLS = 4096
DEFAULT_NPROBE = 1
DEFAULT_FAST_MARGIN = 0.4


class VectorIndex:
    def __init__(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
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
        self.vectors = vectors.astype(np.uint8, copy=False)
        self.labels = labels.astype(np.uint8, copy=False)
        self.centroids = centroids.astype(np.float32, copy=False) if centroids is not None else None
        self.offsets = offsets.astype(np.int64, copy=False) if offsets is not None else None
        self.block_size = block_size
        self.nprobe = max(1, nprobe)
        self.fast_margin = float(os.getenv("RINHA_CELL_FAST_MARGIN", str(DEFAULT_FAST_MARGIN)))
        self.cell_scores = self._build_cell_scores()

    def score(self, query: np.ndarray) -> float:
        if self.labels.shape[0] == 0:
            return 1.0
        if self.centroids is not None and self.offsets is not None:
            return self._score_ivf(query)
        return self._score_exact(query)

    def _score_exact(self, query: np.ndarray) -> float:
        neighbors = min(5, self.labels.shape[0])
        quantized = quantize_vector(query).astype(np.int16, copy=False)
        best_distances = np.full(neighbors, np.iinfo(np.int32).max, dtype=np.int32)
        best_labels = np.zeros(neighbors, dtype=np.uint8)

        for start in range(0, self.labels.shape[0], self.block_size):
            end = min(start + self.block_size, self.labels.shape[0])
            block = self.vectors[start:end].astype(np.int16, copy=False)
            diff = block - quantized
            distances = np.sum(diff.astype(np.int32, copy=False) * diff.astype(np.int32, copy=False), axis=1)
            take = min(neighbors, distances.shape[0])
            local = np.argpartition(distances, take - 1)[:take]

            merged_distances = np.concatenate((best_distances, distances[local]))
            merged_labels = np.concatenate((best_labels, self.labels[start:end][local]))
            keep = np.argpartition(merged_distances, neighbors - 1)[:neighbors]
            best_distances = merged_distances[keep]
            best_labels = merged_labels[keep]

        frauds = int(np.sum(best_labels))
        return frauds / float(neighbors)

    def _score_ivf(self, query: np.ndarray) -> float:
        assert self.centroids is not None
        assert self.offsets is not None

        probes = min(self.nprobe, self.centroids.shape[0])
        center_distances = np.sum((self.centroids - query.astype(np.float32, copy=False)) ** 2, axis=1)
        if self.cell_scores is not None:
            nearest = int(np.argmin(center_distances))
            cell_score = float(self.cell_scores[nearest])
            if cell_score <= 0.5 - self.fast_margin or cell_score >= 0.5 + self.fast_margin:
                return cell_score
        cells = np.argpartition(center_distances, probes - 1)[:probes]
        quantized = quantize_vector(query).astype(np.int16, copy=False)

        neighbors = min(5, self.labels.shape[0])
        best_distances = np.full(neighbors, np.iinfo(np.int32).max, dtype=np.int32)
        best_labels = np.zeros(neighbors, dtype=np.uint8)
        found = 0

        for cell in cells:
            cell_start = int(self.offsets[cell])
            cell_end = int(self.offsets[cell + 1])
            if cell_start == cell_end:
                continue

            for start in range(cell_start, cell_end, self.block_size):
                end = min(start + self.block_size, cell_end)
                block = self.vectors[start:end].astype(np.int16, copy=False)
                diff = block - quantized
                distances = np.sum(diff.astype(np.int32, copy=False) * diff.astype(np.int32, copy=False), axis=1)
                take = min(neighbors, distances.shape[0])
                local = np.argpartition(distances, take - 1)[:take]

                merged_distances = np.concatenate((best_distances, distances[local]))
                merged_labels = np.concatenate((best_labels, self.labels[start:end][local]))
                keep = np.argpartition(merged_distances, neighbors - 1)[:neighbors]
                best_distances = merged_distances[keep]
                best_labels = merged_labels[keep]
                found += take

        if found == 0:
            return self._score_exact(query)

        usable = min(neighbors, found)
        best = np.argpartition(best_distances, usable - 1)[:usable]
        frauds = int(np.sum(best_labels[best]))
        return frauds / float(usable)

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


def load_index(index_dir: Path) -> VectorIndex:
    quantized_path = index_dir / QUANTIZED_FILE
    if quantized_path.exists():
        vectors = np.load(quantized_path, mmap_mode="r")
    else:
        vectors = quantize_vectors(np.load(index_dir / VECTORS_FILE, mmap_mode="r"))
    labels = np.load(index_dir / LABELS_FILE, mmap_mode="r")
    centroids = None
    offsets = None
    if (index_dir / CENTROIDS_FILE).exists() and (index_dir / OFFSETS_FILE).exists():
        centroids = np.load(index_dir / CENTROIDS_FILE, mmap_mode="r")
        offsets = np.load(index_dir / OFFSETS_FILE, mmap_mode="r")
    nprobe = int(os.getenv("RINHA_IVF_NPROBE", str(DEFAULT_NPROBE)))
    return VectorIndex(vectors, labels, centroids=centroids, offsets=offsets, nprobe=nprobe)


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
    np.save(index_dir / LABELS_FILE, labels[order])
    np.save(index_dir / CENTROIDS_FILE, centroids)
    np.save(index_dir / OFFSETS_FILE, offsets)

    stat = references_path.stat()
    meta = {
        "algorithm": "ivf-flat",
        "dimensions": 14,
        "count": len(records),
        "cells": cell_count,
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


def index_matches_source(index_dir: Path, references_path: Path) -> bool:
    required = [QUANTIZED_FILE, LABELS_FILE, CENTROIDS_FILE, OFFSETS_FILE, META_FILE]
    if any(not (index_dir / filename).exists() for filename in required):
        return False

    try:
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        stat = references_path.stat()
    except (OSError, ValueError):
        return False

    return (
        meta.get("algorithm") == "ivf-flat"
        and meta.get("cells") == int(os.getenv("RINHA_IVF_CELLS", str(DEFAULT_CELLS)))
        and meta.get("dimensions") == 14
        and meta.get("source_name") == references_path.name
        and meta.get("source_size") == stat.st_size
        and meta.get("source_mtime_ns") == stat.st_mtime_ns
    )
