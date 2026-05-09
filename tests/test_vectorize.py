from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION, parse_utc_timestamp_metrics, vectorize_payload


def test_vectorizes_payload_from_rules_example() -> None:
    payload = {
        "id": "tx-1329056812",
        "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
        "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
        "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 29.23},
        "last_transaction": None,
    }

    vector = vectorize_payload(payload, DEFAULT_NORMALIZATION, DEFAULT_MCC_RISK)

    assert vector.tolist() == [
        0.004112000111490488,
        0.1666666716337204,
        0.05000000074505806,
        0.782608687877655,
        0.3333333432674408,
        -1.0,
        -1.0,
        0.02923000045120716,
        0.15000000596046448,
        0.0,
        1.0,
        0.0,
        0.15000000596046448,
        0.006025000009685755,
    ]


def test_fast_timestamp_parser_matches_datetime_fallback() -> None:
    seconds, hour, weekday = parse_utc_timestamp_metrics("2026-03-11T18:45:53Z")

    assert seconds == 1773254753.0
    assert hour == 18
    assert weekday == 2
