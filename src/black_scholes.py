"""
Black-Scholes European option pricer (analytic).

Why a class with const-style methods (no mutation)?
  The five market parameters (S, K, r, σ, T) are fixed for a contract
  snapshot. Bundling them keeps call/put/Greeks consistent with the same
  inputs and makes the API read like the math:

      bs = BlackScholes(S, K, r, sigma, T)
      c = bs.call_price()

Why separate Greek methods?
  Each Greek is its own closed-form expression. Independent methods make
  that obvious, avoid forcing a "return everything" struct when you only
  need one number, and guarantee they never mutate stored parameters.

Part 1 scope: vanilla European call/put + analytic Greeks.
Later: implied vol, dividends, American approx, Monte Carlo, backtests.
"""

from __future__ import annotations

import math


class BlackScholes:
    def __init__(
        self,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        time_to_expiry: float,
    ) -> None:
        """
        Parameters
        ----------
        spot : float
            Current underlying price S > 0
        strike : float
            Strike price K > 0
        rate : float
            Continuously compounded risk-free rate r
        vol : float
            Annualized volatility σ > 0
        time_to_expiry : float
            Time to expiry in years T > 0
        """
        if spot <= 0.0 or strike <= 0.0 or vol <= 0.0 or time_to_expiry <= 0.0:
            raise ValueError(
                "BlackScholes requires S > 0, K > 0, sigma > 0, and T > 0"
            )

        self._S = float(spot)
        self._K = float(strike)
        self._r = float(rate)
        self._sigma = float(vol)
        self._T = float(time_to_expiry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _d1(self) -> float:
        """d1 = [ln(S/K) + (r + σ²/2) T] / (σ √T)"""
        return (
            math.log(self._S / self._K)
            + (self._r + 0.5 * self._sigma * self._sigma) * self._T
        ) / (self._sigma * math.sqrt(self._T))

    def _d2(self) -> float:
        """d2 = d1 - σ √T"""
        return self._d1() - self._sigma * math.sqrt(self._T)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """
        Standard normal CDF Φ(x).

        Uses math.erfc for accuracy (complementary error function), not a
        hand-rolled polynomial approximation.
        Φ(x) = ½ erfc(-x / √2)
        """
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF φ(x)."""
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def call_price(self) -> float:
        """European call: C = S N(d1) - K e^{-rT} N(d2)"""
        d1 = self._d1()
        d2 = self._d2()
        return self._S * self._norm_cdf(d1) - self._K * math.exp(
            -self._r * self._T
        ) * self._norm_cdf(d2)

    def put_price(self) -> float:
        """European put: P = K e^{-rT} N(-d2) - S N(-d1)"""
        d1 = self._d1()
        d2 = self._d2()
        return self._K * math.exp(-self._r * self._T) * self._norm_cdf(
            -d2
        ) - self._S * self._norm_cdf(-d1)

    # ------------------------------------------------------------------
    # Greeks (closed-form; no numerical differentiation)
    # ------------------------------------------------------------------

    def delta(self, is_call: bool) -> float:
        """
        Delta ∂V/∂S
          Call: N(d1)
          Put:  N(d1) - 1
        """
        nd1 = self._norm_cdf(self._d1())
        return nd1 if is_call else nd1 - 1.0

    def gamma(self) -> float:
        """Gamma ∂²V/∂S² (same for call and put): n(d1) / (S σ √T)"""
        return self._norm_pdf(self._d1()) / (
            self._S * self._sigma * math.sqrt(self._T)
        )

    def vega(self) -> float:
        """
        Vega ∂V/∂σ (same for call and put): S n(d1) √T

        Returned in absolute units (not "per 1% vol").
        """
        return self._S * self._norm_pdf(self._d1()) * math.sqrt(self._T)

    def theta(self, is_call: bool) -> float:
        """
        Theta ∂V/∂T (per year of calendar time)
          Call: -S n(d1) σ / (2√T) - r K e^{-rT} N(d2)
          Put:  -S n(d1) σ / (2√T) + r K e^{-rT} N(-d2)
        """
        sqrt_t = math.sqrt(self._T)
        d1 = self._d1()
        d2 = self._d2()
        discount = math.exp(-self._r * self._T)
        time_decay = -self._S * self._norm_pdf(d1) * self._sigma / (2.0 * sqrt_t)

        if is_call:
            return time_decay - self._r * self._K * discount * self._norm_cdf(d2)
        return time_decay + self._r * self._K * discount * self._norm_cdf(-d2)

    def rho(self, is_call: bool) -> float:
        """
        Rho ∂V/∂r
          Call:  K T e^{-rT} N(d2)
          Put:  -K T e^{-rT} N(-d2)
        """
        discount = math.exp(-self._r * self._T)
        if is_call:
            return self._K * self._T * discount * self._norm_cdf(self._d2())
        return -self._K * self._T * discount * self._norm_cdf(-self._d2())
