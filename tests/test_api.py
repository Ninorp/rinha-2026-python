from types import SimpleNamespace

import msgspec

from rinha_api import app as api


class StubIndex:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, query) -> float:
        return self._score


def payload() -> dict:
    return {
        "id": "tx-1329056812",
        "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
        "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
        "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 29.23},
        "last_transaction": None,
    }


def test_ready_returns_200_when_initialized() -> None:
    response = api.ready()

    assert response.status_code == 200
    assert response.description == "ok"


def test_fraud_score_approves_when_score_is_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(api, "_index", StubIndex(0.4))
    body = msgspec.json.encode(payload())

    response = api.fraud_score(SimpleNamespace(body=body))

    assert response.status_code == 200
    assert msgspec.json.decode(response.description) == {"approved": True, "fraud_score": 0.4}


def test_fraud_score_denies_when_score_reaches_threshold(monkeypatch) -> None:
    monkeypatch.setattr(api, "_index", StubIndex(0.6))
    body = msgspec.json.encode(payload())

    response = api.fraud_score(SimpleNamespace(body=body))

    assert response.status_code == 200
    assert msgspec.json.decode(response.description) == {"approved": False, "fraud_score": 0.6}
