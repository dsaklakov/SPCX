import math
import numpy as np


def clamp(x, low=0, high=100):
    return max(low, min(high, x))


def score_by_thresholds(value, thresholds, scores):
    for threshold, score in zip(thresholds, scores):
        if value <= threshold:
            return score
    return scores[-1]


def score_valuation(current_price, fair_value):
    if fair_value <= 0:
        return 0

    premium = (
        (current_price - fair_value)
        / fair_value
        * 100
    )

    return score_by_thresholds(
        premium,
        [34, 55, 89, 144, 233],
        [20, 40, 60, 80, 100],
    )


def score_expected_upside(
    current_price,
    targets,
    probabilities
):
    if current_price <= 0:
        return 0

    ev = 0

    for target, prob in zip(targets, probabilities):
        if target > current_price:
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
    elif momentum < 144:
        return 100
    else:
        return 100


def score_price_acceleration(return_1d, return_3d, return_5d):
    medium_return = max(abs(return_5d), abs(return_3d), 0.01)
    acceleration_ratio = abs(return_1d) / medium_return

    return score_by_thresholds(
        acceleration_ratio,
        [0.5, 1.0, 1.5, 2.5, 4.0],
        [20, 40, 60, 80, 100],
    )


def score_volatility_ratio(volatility_ratio):
    if volatility_ratio <= 0:
        return 0

    return score_by_thresholds(
        volatility_ratio,
        [1.0, 1.5, 2.0, 3.0, 5.0],
        [20, 40, 60, 80, 100],
    )


def score_liquidity_climax(current_price, day_low, day_high, current_volume, avg_volume_20d):
    if day_high <= day_low or avg_volume_20d <= 0:
        return 0

    price_position = (current_price - day_low) / (day_high - day_low)

    if price_position < 0.8:
        return 0

    volume_ratio = current_volume / avg_volume_20d

    return score_by_thresholds(
        volume_ratio,
        [2, 3, 5, 8, 13],
        [20, 40, 60, 80, 100],
    )


def score_position_size(position_value, total_portfolio_value):
    if total_portfolio_value <= 0:
        return 0

    position_pct = position_value / total_portfolio_value * 100

    return score_by_thresholds(
        position_pct,
        [5, 10, 15, 25, 40],
        [20, 40, 60, 80, 100],
    )


def score_time_horizon(days_until_money_needed):
    if days_until_money_needed < 0:
        days_until_money_needed = 0

    return clamp(100 * math.exp(-days_until_money_needed / 90.0))


def score_portfolio_concentration(position_value, total_portfolio_value, second_largest_position):
    if total_portfolio_value <= 0:
        concentration_score = 0
    else:
        position_pct = position_value / total_portfolio_value * 100
        concentration_score = score_by_thresholds(
            position_pct,
            [10, 20, 30, 50, 80],
            [20, 40, 60, 80, 100],
        )

    if second_largest_position <= 0:
        ratio_score = 100
    else:
        position_ratio = position_value / second_largest_position
        ratio_score = score_by_thresholds(
            position_ratio,
            [1.5, 2, 3, 5, 8],
            [20, 40, 60, 80, 100],
        )

    return 0.6 * concentration_score + 0.4 * ratio_score


def score_event_risk(severity, probability, days_to_event):
    severity = clamp(severity, 0, 10)
    probability = clamp(probability, 0, 1)
    days_to_event = max(days_to_event, 0)

    event_raw = 10 * severity * probability * math.exp(-days_to_event / 30.0)
    return clamp(event_raw)


def score_narrative_risk(manual_score=None, media_ratio=None):
    if manual_score is not None:
        return clamp(manual_score)

    if media_ratio is None or media_ratio <= 0:
        return 0

    return score_by_thresholds(
        media_ratio,
        [2, 3, 5, 8, 13],
        [20, 40, 60, 80, 100],
    )


def score_execution_risk(
    starship_status_risk,
    launch_cadence_risk,
    nasa_dependence_risk,
    defense_contracts_risk,
):
    return clamp(
        np.mean(
            [
                clamp(starship_status_risk),
                clamp(launch_cadence_risk),
                clamp(nasa_dependence_risk),
                clamp(defense_contracts_risk),
            ]
        )
    )
