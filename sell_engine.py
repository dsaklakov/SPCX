import numpy as np


def clamp(x, low=0, high=100):
    return max(low, min(high, x))


def score_valuation(current_price, fair_value):
    if fair_value <= 0:
        return 0

    premium = (
        (current_price - fair_value)
        / fair_value
        * 100
    )

    if premium <= 34:
        return 20
    elif premium <= 55:
        return 40
    elif premium <= 89:
        return 60
    elif premium <= 144:
        return 80
    else:
        return 100


def score_expected_upside(
    current_price,
    targets,
    probabilities
):
    ev = 0

    for target, prob in zip(targets, probabilities):
        ev += prob * (target - current_price)

    ev_pct = ev / current_price * 100

    if ev_pct < -10:
        return 100

    elif ev_pct < 0:
        return 80

    elif ev_pct < 5:
        return 60

    elif ev_pct < 10:
        return 40

    else:
        return 20


def score_momentum(
    current_price,
    price_5d
):
    if price_5d <= 0:
        return 0

    momentum = (
        (current_price - price_5d)
        / price_5d
        * 100
    )

    if momentum < 21:
        return 20
    elif momentum < 34:
        return 40
    elif momentum < 55:
        return 60
    elif momentum < 89:
        return 80
    else:
        return 100
