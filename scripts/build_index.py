from __future__ import annotations

import argparse
from pathlib import Path

from rinha_api.index import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", default="resources/references.json.gz")
    parser.add_argument("--out", default="resources/index")
    parser.add_argument("--cells", type=int, default=None)
    args = parser.parse_args()

    build_index(Path(args.references), Path(args.out), cells=args.cells)


if __name__ == "__main__":
    main()
