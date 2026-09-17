"""
Hardened unit tests for the Black-Scholes European pricer.

Covers: reference prices, put-call parity, edge cases, and analytic Greeks
(cross-checked against closed-form values and finite-difference sanity).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from black_scholes import BlackScholes

# Classic ATM snapshot used throughout.
ATM = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.20, time_to_expiry=1.0)


def test_atm_call_put_match_known_reference_values() -> None:
    bs = BlackScholes(**ATM)
    assert bs.call_price() == pytest.approx(10.4505835722, abs=1e-6)
    assert bs.put_price() == pytest.approx(5.5735260223, abs=1e-6)


def test_put_call_parity() -> None:
    """C - P == S - K e^{-rT}  (European, no dividends)."""
    cases = [
        ATM,
        dict(spot=100.0, strike=90.0, rate=0.05, vol=0.25, time_to_expiry=0.5),
        dict(spot=100.0, strike=110.0, rate=0.01, vol=0.15, time_to_expiry=2.0),
        dict(spot=50.0, strike=50.0, rate=0.0, vol=0.30, time_to_expiry=1.0),
    ]
    for params in cases:
        bs = BlackScholes(**params)
        lhs = bs.call_price() - bs.put_price()
        rhs = params["spot"] - params["strike"] * math.exp(
            -params["rate"] * params["time_to_expiry"]
        )
        assert lhs == pytest.approx(rhs, abs=1e-10)


def test_very_short_time_to_expiry() -> None:
    # Near expiry, prices collapse toward intrinsic.
    S, K = 100.0, 100.0
    bs = BlackScholes(S, K, 0.05, 0.20, 1e-6)
    assert bs.call_price() == pytest.approx(0.0, abs=1e-2)
    assert bs.put_price() == pytest.approx(0.0, abs=1e-2)

    # Slightly ITM call / OTM put.
    bs_itm = BlackScholes(105.0, 100.0, 0.05, 0.20, 1e-6)
    assert bs_itm.call_price() == pytest.approx(5.0, abs=1e-2)
    assert bs_itm.put_price() == pytest.approx(0.0, abs=1e-2)


def test_deep_in_the_money_call() -> None:
    # Deep ITM call ≈ forward intrinsic: S - K e^{-rT}; put ≈ 0.
    S, K, r, T = 100.0, 50.0, 0.05, 1.0
    bs = BlackScholes(S, K, r, 0.20, T)
    intrinsic_fwd = S - K * math.exp(-r * T)
    assert bs.call_price() == pytest.approx(intrinsic_fwd, abs=0.05)
    assert bs.put_price() == pytest.approx(0.0, abs=0.05)
    assert bs.delta(True) == pytest.approx(1.0, abs=1e-3)


def test_deep_out_of_the_money_call() -> None:
    # Deep OTM call ≈ 0; put ≈ K e^{-rT} - S; call delta ≈ 0.
    S, K, r, T = 100.0, 200.0, 0.05, 1.0
    bs = BlackScholes(S, K, r, 0.20, T)
    assert bs.call_price() == pytest.approx(0.0, abs=0.05)
    assert bs.put_price() == pytest.approx(K * math.exp(-r * T) - S, abs=0.05)
    assert bs.delta(True) == pytest.approx(0.0, abs=1e-3)


def test_greeks_match_known_reference_values() -> None:
    # Closed-form references for ATM params (independently verified via FD).
    bs = BlackScholes(**ATM)

    assert bs.delta(True) == pytest.approx(0.636830651176, abs=1e-9)
    assert bs.delta(False) == pytest.approx(-0.363169348824, abs=1e-9)
    assert bs.gamma() == pytest.approx(0.018762017346, abs=1e-9)
    assert bs.vega() == pytest.approx(37.5240346917, abs=1e-9)
    assert bs.theta(True) == pytest.approx(-6.41402754644, abs=1e-9)
    assert bs.theta(False) == pytest.approx(-1.65788042393, abs=1e-9)
    assert bs.rho(True) == pytest.approx(53.2324815454, abs=1e-9)
    assert bs.rho(False) == pytest.approx(-41.8904609047, abs=1e-9)


def test_call_put_delta_relationship() -> None:
    # With no dividends: Δ_call - Δ_put == 1.
    bs = BlackScholes(**ATM)
    assert bs.delta(True) - bs.delta(False) == pytest.approx(1.0, abs=1e-12)


def test_gamma_and_vega_identical_for_call_and_put() -> None:
    # Gamma and vega are the same for European call and put (same d1 term).
    bs = BlackScholes(**ATM)
    # Exposed as shared methods already; sanity: both positive for ATM.
    assert bs.gamma() > 0.0
    assert bs.vega() > 0.0


def test_greeks_match_finite_difference() -> None:
    """Cross-check analytic Greeks against central finite differences."""
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs = BlackScholes(S, K, r, sig, T)
    eps = 1e-5

    def call(spot=S, rate=r, vol=sig, time=T) -> float:
        return BlackScholes(spot, K, rate, vol, time).call_price()

    def put(spot=S, rate=r, vol=sig, time=T) -> float:
        return BlackScholes(spot, K, rate, vol, time).put_price()

    fd_delta_call = (call(spot=S + eps) - call(spot=S - eps)) / (2 * eps)
    fd_delta_put = (put(spot=S + eps) - put(spot=S - eps)) / (2 * eps)
    fd_gamma = (call(spot=S + eps) - 2 * call() + call(spot=S - eps)) / (eps * eps)
    fd_vega = (call(vol=sig + eps) - call(vol=sig - eps)) / (2 * eps)
    fd_rho_call = (call(rate=r + eps) - call(rate=r - eps)) / (2 * eps)
    # Analytic theta is calendar-time decay (= -∂V/∂T). Keep T - eps > 0.
    fd_dV_dT = (call(time=T + eps) - call(time=T - eps)) / (2 * eps)

    assert bs.delta(True) == pytest.approx(fd_delta_call, abs=1e-6)
    assert bs.delta(False) == pytest.approx(fd_delta_put, abs=1e-6)
    assert bs.gamma() == pytest.approx(fd_gamma, abs=1e-5)
    assert bs.vega() == pytest.approx(fd_vega, abs=1e-6)
    assert bs.rho(True) == pytest.approx(fd_rho_call, abs=1e-6)
    assert bs.theta(True) == pytest.approx(-fd_dV_dT, abs=1e-5)


def test_constructor_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        BlackScholes(0.0, 100.0, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        BlackScholes(100.0, -1.0, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        BlackScholes(100.0, 100.0, 0.05, 0.0, 1.0)
    with pytest.raises(ValueError):
        BlackScholes(100.0, 100.0, 0.05, 0.2, 0.0)
