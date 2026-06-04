from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rinha_api.config import index_dir, load_json_or_default, resources_dir
from rinha_api.index import load_index, quantize_vector
from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION, vectorize_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="../load-test/test-data.json")
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    resources = resources_dir()
    index = load_index(index_dir())
    index.nprobe = args.nprobe
    index.tree = None

    normalization = load_json_or_default(resources / "normalization.json", DEFAULT_NORMALIZATION)
    mcc_risk = load_json_or_default(resources / "mcc_risk.json", DEFAULT_MCC_RISK)
    entries = json.loads(Path(args.test_data).read_bytes())["entries"]

    disagreements = []
    shortcut_count = 0
    shortcut_errors = 0
    for position, entry in enumerate(entries):
        query = vectorize_payload(entry["request"], normalization, mcc_risk)
        metrics = cell_metrics(index, query)
        if metrics is None:
            continue
        shortcut_count += 1

        fast_score = metrics["cell_score"]
        fast_approved = fast_score < 0.6
        expected = bool(entry["expected_approved"])
        if fast_approved != expected:
            shortcut_errors += 1

        original_margin = index.fast_margin
        index.fast_margin = 999.0
        full_score = index.score(query)
        index.fast_margin = original_margin
        full_approved = full_score < 0.6

        if fast_approved != full_approved:
            disagreements.append(
                {
                    "position": position,
                    "expected": expected,
                    "fast_score": fast_score,
                    "full_score": full_score,
                    **metrics,
                }
            )

    print(f"shortcuts={shortcut_count} shortcut_errors={shortcut_errors} disagreements={len(disagreements)}")
    for item in sorted(disagreements, key=lambda row: row["center_gap"])[: args.top]:
        print(
            "pos={position} expected={expected} fast={fast_score:.1f} full={full_score:.1f} "
            "cell={cell} size={cell_size} gap={bound_gap} next_lb={next_lower_bound} "
            "nearest_lb={nearest_lower_bound} center_gap={center_gap:.6f}".format(**item)
        )


def cell_metrics(index, query: np.ndarray) -> dict | None:
    if index.centroids is None or index.offsets is None or index.centroid_norms is None:
        return None
    if index.cell_scores is None:
        return None

    query_f32 = query.astype(np.float32, copy=False)
    query_norm = float(np.sum(query_f32 * query_f32))
    center_distances = index.centroid_norms + query_norm - 2.0 * (index.centroids @ query_f32)
    nearest = int(np.argmin(center_distances))
    cell_score = float(index.cell_scores[nearest])
    if not (cell_score <= 0.5 - index.fast_margin or cell_score >= 0.5 + index.fast_margin):
        return None

    nearest_lb = 0
    next_lower_bound = 0
    if index.bounds is not None:
        quantized = quantize_vector(query).astype(np.int16, copy=False)
        nearest_lb = int(index._cell_lower_bounds(np.array([nearest]), quantized)[0])
        probes = min(max(index.nprobe, 2), index.centroids.shape[0])
        cells = np.argpartition(center_distances, probes - 1)[:probes]
        lower_bounds = index._cell_lower_bounds(cells, quantized)
        order = np.argsort(lower_bounds, kind="stable")
        sorted_cells = cells[order]
        sorted_bounds = lower_bounds[order]
        next_lower_bound = None
        for cell, lower_bound in zip(sorted_cells, sorted_bounds):
            if int(cell) != nearest:
                next_lower_bound = int(lower_bound)
                break
        if next_lower_bound is None:
            next_lower_bound = nearest_lb

    center_order = np.argpartition(center_distances, 1)[:2]
    center_gap = float(np.max(center_distances[center_order]) - np.min(center_distances[center_order]))
    start = int(index.offsets[nearest])
    end = int(index.offsets[nearest + 1])
    return {
        "cell": nearest,
        "cell_score": cell_score,
        "cell_size": end - start,
        "nearest_lower_bound": nearest_lb,
        "next_lower_bound": next_lower_bound,
        "bound_gap": next_lower_bound - nearest_lb,
        "center_gap": center_gap,
    }


if __name__ == "__main__":
    main()
