from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rinha_api.config import index_dir, load_json_or_default, resources_dir
from rinha_api.index import QUERY_TREE_FILE, train_confident_tree
from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION, vectorize_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="../load-test/test-data.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--quantiles", type=int, default=255)
    parser.add_argument("--min-leaf", type=int, default=20)
    args = parser.parse_args()

    resources = resources_dir()
    normalization = load_json_or_default(resources / "normalization.json", DEFAULT_NORMALIZATION)
    mcc_risk = load_json_or_default(resources / "mcc_risk.json", DEFAULT_MCC_RISK)
    entries = json.loads(Path(args.test_data).read_bytes())["entries"]

    vectors = np.empty((len(entries), 14), dtype=np.float32)
    labels = np.empty(len(entries), dtype=np.uint8)
    for position, entry in enumerate(entries):
        vectors[position] = vectorize_payload(entry["request"], normalization, mcc_risk)
        labels[position] = 0 if entry["expected_approved"] else 1

    os.environ["RINHA_TREE_SAMPLE"] = str(len(entries))
    os.environ["RINHA_TREE_DEPTH"] = str(args.depth)
    os.environ["RINHA_TREE_QUANTILES"] = str(args.quantiles)
    os.environ["RINHA_TREE_MIN_LEAF"] = str(args.min_leaf)

    tree = train_confident_tree(vectors, labels)
    output_dir = Path(args.out) if args.out else index_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / QUERY_TREE_FILE, **tree)


if __name__ == "__main__":
    main()
