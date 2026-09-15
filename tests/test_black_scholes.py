"""
Part 1 smoke tests — prove the core formulas match known references.

Part 2 expands coverage (put-call parity, edges, every Greek vs references).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from black_scholes import BlackScholes


def test_atm_call_put_match_known_reference_values() -> None:
    # S=100, K=100, r=5%, σ=20%, T=1y — call ≈ 10.4506 (textbook check).
    bs = BlackScholes(100.0, 100.0, 0.05, 0.20, 1.0)
    assert bs.call_price() == pytest.approx(10.4505835722, abs=1e-6)
    assert bs.put_price() == pytest.approx(5.5735260223, abs=1e-6)


def test_constructor_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        BlackScholes(0.0, 100.0, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        BlackScholes(100.0, -1.0, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        BlackScholes(100.0, 100.0, 0.05, 0.0, 1.0)
    with pytest.raises(ValueError):
        BlackScholes(100.0, 100.0, 0.05, 0.2, 0.0)
