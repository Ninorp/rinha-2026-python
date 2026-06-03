from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from rinha_api.config import load_json_or_default, resources_dir
from rinha_api.index import build_quantized_cell_bounds, load_index
from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION, vectorize_payload


def score_entries(index, vectors, cell_prune: bool, batch_cells: bool = False) -> tuple[list[float], float]:
    index.cell_prune = cell_prune
    index.batch_cells = batch_cells
    index.probe_counts = [] if cell_prune else None
    started = perf_counter()
    scores = [index.score(vector) for vector in vectors]
    return scores, perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="../load-test/test-data.json")
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--baseline-cell-prune", action="store_true")
    parser.add_argument("--compare-batch-cells", action="store_true")
    parser.add_argument("--disable-tree", action="store_true")
    parser.add_argument("--disable-cell-fast-path", action="store_true")
    args = parser.parse_args()

    resources = resources_dir()
    index = load_index(resources / "index")
    if index.bounds is None:
        if index.centroids is None or index.offsets is None:
            raise SystemExit("index has no IVF cell bounds")
        index.bounds = build_quantized_cell_bounds(index.vectors, index.centroids, index.offsets)
    index.nprobe = args.nprobe
    if args.disable_tree:
        index.tree = None
    if args.disable_cell_fast_path:
        index.fast_margin = 1.0

    normalization = load_json_or_default(resources / "normalization.json", DEFAULT_NORMALIZATION)
    mcc_risk = load_json_or_default(resources / "mcc_risk.json", DEFAULT_MCC_RISK)
    entries = json.loads(Path(args.test_data).read_bytes())["entries"]
    vectors = [vectorize_payload(entry["request"], normalization, mcc_risk) for entry in entries]

    baseline, baseline_seconds = score_entries(index, vectors, cell_prune=args.baseline_cell_prune)
    pruned, pruned_seconds = score_entries(index, vectors, cell_prune=True, batch_cells=args.compare_batch_cells)
    score_changes = sum(left != right for left, right in zip(baseline, pruned))
    approval_changes = sum((left < 0.6) != (right < 0.6) for left, right in zip(baseline, pruned))
    probe_counts = index.probe_counts or []
    histogram = Counter(probe_counts)
    average = sum(probe_counts) / len(probe_counts) if probe_counts else 0.0

    print(
        f"entries={len(entries)} nprobe={args.nprobe} "
        f"tree={'off' if args.disable_tree else 'on'} "
        f"baseline_cell_prune={'on' if args.baseline_cell_prune else 'off'} "
        f"cell_fast={'off' if args.disable_cell_fast_path else 'on'} "
        f"batch_cells={'on' if args.compare_batch_cells else 'off'}"
    )
    print(
        f"baseline={baseline_seconds:.3f}s pruned={pruned_seconds:.3f}s "
        f"score_changes={score_changes} approval_changes={approval_changes}"
    )
    print(
        f"ivf_fallbacks={len(probe_counts)} average_visited={average:.3f} "
        f"histogram={dict(sorted(histogram.items()))}"
    )
    if score_changes or approval_changes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
