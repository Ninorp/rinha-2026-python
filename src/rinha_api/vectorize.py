from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


DEFAULT_MCC_RISK = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}

DEFAULT_NORMALIZATION = {
    "max_amount": 10000.0,
    "max_installments": 12.0,
    "amount_vs_avg_ratio": 10.0,
    "max_minutes": 1440.0,
    "max_km": 1000.0,
    "max_tx_count_24h": 20.0,
    "max_merchant_avg_amount": 10000.0,
}


def clamp(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def parse_utc_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def vectorize_payload(
    payload: dict[str, Any],
    normalization: dict[str, float],
    mcc_risk: dict[str, float],
) -> np.ndarray:
    transaction = payload["transaction"]
    customer = payload["customer"]
    merchant = payload["merchant"]
    terminal = payload["terminal"]
    last_transaction = payload.get("last_transaction")

    amount = float(transaction["amount"])
    customer_avg = float(customer["avg_amount"])
    requested_at = parse_utc_timestamp(transaction["requested_at"])

    vector = np.empty(14, dtype=np.float32)
    vector[0] = clamp(amount / normalization["max_amount"])
    vector[1] = clamp(float(transaction["installments"]) / normalization["max_installments"])
    vector[2] = clamp((amount / customer_avg) / normalization["amount_vs_avg_ratio"]) if customer_avg > 0 else 1.0
    vector[3] = requested_at.hour / 23.0
    vector[4] = requested_at.weekday() / 6.0

    if last_transaction is None:
        vector[5] = -1.0
        vector[6] = -1.0
    else:
        previous_at = parse_utc_timestamp(last_transaction["timestamp"])
        minutes = max(0.0, (requested_at - previous_at).total_seconds() / 60.0)
        vector[5] = clamp(minutes / normalization["max_minutes"])
        vector[6] = clamp(float(last_transaction["km_from_current"]) / normalization["max_km"])

    vector[7] = clamp(float(terminal["km_from_home"]) / normalization["max_km"])
    vector[8] = clamp(float(customer["tx_count_24h"]) / normalization["max_tx_count_24h"])
    vector[9] = 1.0 if terminal["is_online"] else 0.0
    vector[10] = 1.0 if terminal["card_present"] else 0.0
    vector[11] = 0.0 if merchant["id"] in customer["known_merchants"] else 1.0
    vector[12] = float(mcc_risk.get(str(merchant["mcc"]), 0.5))
    vector[13] = clamp(float(merchant["avg_amount"]) / normalization["max_merchant_avg_amount"])
    return vector
