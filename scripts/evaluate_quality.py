from __future__ import annotations

import argparse
import json
from pathlib import Path

from rinha_api.config import load_json_or_default, resources_dir
from rinha_api.index import load_index
from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION, vectorize_payload


def evaluate(
    test_data_path: Path,
    nprobe: int,
    disable_tree: bool,
    show_errors: bool,
) -> tuple[int, int, int, int, int]:
    resources = resources_dir()
    index = load_index(resources / "index")
    index.nprobe = nprobe
    if disable_tree:
        index.tree = None

    normalization = load_json_or_default(resources / "normalization.json", DEFAULT_NORMALIZATION)
    mcc_risk = load_json_or_default(resources / "mcc_risk.json", DEFAULT_MCC_RISK)
    entries = json.loads(test_data_path.read_bytes())["entries"]

    fp = fn = tp = tn = 0
    for position, entry in enumerate(entries):
        score = index.score(vectorize_payload(entry["request"], normalization, mcc_risk))
        approved = score < 0.6
        expected = entry["expected_approved"]
        if approved and not expected:
            fn += 1
            if show_errors:
                print_error(position, "fn", score, entry)
        elif not approved and expected:
            fp += 1
            if show_errors:
                print_error(position, "fp", score, entry)
        elif approved:
            tn += 1
        else:
            tp += 1

    return fp, fn, tp, tn, fp + fn * 3


def print_error(position: int, kind: str, score: float, entry: dict) -> None:
    request = entry["request"]
    transaction = request["transaction"]
    customer = request["customer"]
    merchant = request["merchant"]
    terminal = request["terminal"]
    print(
        f"{position} {kind} score={score} expected_score={entry['expected_fraud_score']} "
        f"id={request['id']} amount={transaction['amount']} installments={transaction['installments']} "
        f"tx24={customer['tx_count_24h']} mcc={merchant['mcc']} known={merchant['id'] in customer['known_merchants']} "
        f"online={terminal['is_online']} present={terminal['card_present']} "
        f"km_home={terminal['km_from_home']:.2f} last={request['last_transaction'] is not None}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="../load-test/test-data.json")
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1])
    parser.add_argument("--disable-tree", action="store_true")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    for nprobe in args.nprobe:
        fp, fn, tp, tn, weighted = evaluate(Path(args.test_data), nprobe, args.disable_tree, args.show_errors)
        print(
            f"nprobe={nprobe} tree={'off' if args.disable_tree else 'on'} "
            f"fp={fp} fn={fn} tp={tp} tn={tn} E={weighted}"
        )


if __name__ == "__main__":
    main()
