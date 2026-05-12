import os

import msgspec
from robyn import Request, Response, Robyn
from robyn.argument_parser import Config

from rinha_api.config import index_dir, load_json_or_default, resources_dir
from rinha_api.index import (
    LABELS_FILE,
    QUANTIZED_FILE,
    VectorIndex,
    empty_index,
    load_index,
)
from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION, vectorize_payload


config = Config()
config.log_level = os.getenv("ROBYN_LOG_LEVEL", "WARN")
config.processes = int(os.getenv("ROBYN_PROCESSES", "1"))
config.workers = int(os.getenv("ROBYN_WORKERS", "4"))

app = Robyn(__file__, config=config)
decoder = msgspec.json.Decoder()
encoder = msgspec.json.Encoder()

_ready = False
_index: VectorIndex | None = None
_normalization: dict[str, float] = DEFAULT_NORMALIZATION
_mcc_risk: dict[str, float] = DEFAULT_MCC_RISK


def initialize() -> None:
    global _ready, _index, _normalization, _mcc_risk

    resources = resources_dir()
    idx_dir = index_dir()
    _normalization = load_json_or_default(resources / "normalization.json", DEFAULT_NORMALIZATION)
    _mcc_risk = load_json_or_default(resources / "mcc_risk.json", DEFAULT_MCC_RISK)

    if (idx_dir / QUANTIZED_FILE).exists() and (idx_dir / LABELS_FILE).exists():
        _index = load_index(idx_dir)
    else:
        _index = empty_index()

    _ready = True


@app.get("/ready")
def ready():
    if not _ready:
        return Response(status_code=503, headers={}, description="starting")
    return Response(status_code=200, headers={}, description="ok")


@app.post("/fraud-score")
def fraud_score(request: Request):
    payload = decoder.decode(request.body)
    query = vectorize_payload(payload, _normalization, _mcc_risk)
    score = _index.score(query) if _index is not None else 1.0
    body = encoder.encode({"approved": score < 0.6, "fraud_score": score}).decode("utf-8")
    return Response(status_code=200, headers={"content-type": "application/json"}, description=body)


initialize()


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
