from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import msgspec
import numpy as np


VECTORS_FILE = "vectors.f32.npy"
QUANTIZED_FILE = "vectors.u1.npy"
LABELS_FILE = "labels.u1.npy"


class VectorIndex:
    def __init__(self, vectors: np.ndarray, labels: np.ndarray, block_size: int = 65536) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != 14:
            raise ValueError("vectors must have shape (n, 14)")
        if labels.ndim != 1 or labels.shape[0] != vectors.shape[0]:
            raise ValueError("labels must have one entry per vector")
        self.vectors = vectors.astype(np.uint8, copy=False)
        self.labels = labels.astype(np.uint8, copy=False)
        self.block_size = block_size

    def score(self, query: np.ndarray) -> float:
        if self.labels.shape[0] == 0:
            return 1.0
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


def load_index(index_dir: Path) -> VectorIndex:
    quantized_path = index_dir / QUANTIZED_FILE
    if quantized_path.exists():
        vectors = np.load(quantized_path, mmap_mode="r")
    else:
        vectors = quantize_vectors(np.load(index_dir / VECTORS_FILE, mmap_mode="r"))
    labels = np.load(index_dir / LABELS_FILE, mmap_mode="r")
    return VectorIndex(vectors, labels)


def build_index(references_path: Path, index_dir: Path) -> None:
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

    np.save(index_dir / QUANTIZED_FILE, quantize_vectors(vectors))
    np.save(index_dir / LABELS_FILE, labels)


def empty_index() -> VectorIndex:
    return VectorIndex(np.empty((0, 14), dtype=np.uint8), np.empty(0, dtype=np.uint8))


def quantize_vectors(vectors: np.ndarray) -> np.ndarray:
    clipped = np.clip(vectors, -1.0, 1.0)
    return np.rint((clipped + 1.0) * 127.5).astype(np.uint8)


def quantize_vector(vector: np.ndarray) -> np.ndarray:
    clipped = np.clip(vector, -1.0, 1.0)
    return np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
