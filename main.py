"""Part 1 demo: print European call/put prices and analytic Greeks."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python main.py` from the repo root without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from black_scholes import BlackScholes


def main() -> None:
    # Classic ATM example used in many textbooks / interview answers.
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs = BlackScholes(S, K, r, sigma, T)

    print("Black-Scholes European option (Part 1 demo)")
    print("------------------------------------------")
    print(f"S={S}  K={K}  r={r}  sigma={sigma}  T={T}\n")

    print(f"Call price: {bs.call_price():.6f}")
    print(f"Put price:  {bs.put_price():.6f}\n")

    print("Greeks")
    print(f"  Delta (call): {bs.delta(True):.6f}")
    print(f"  Delta (put):  {bs.delta(False):.6f}")
    print(f"  Gamma:        {bs.gamma():.6f}")
    print(f"  Vega:         {bs.vega():.6f}")
    print(f"  Theta (call): {bs.theta(True):.6f}")
    print(f"  Theta (put):  {bs.theta(False):.6f}")
    print(f"  Rho (call):   {bs.rho(True):.6f}")
    print(f"  Rho (put):    {bs.rho(False):.6f}")


if __name__ == "__main__":
    main()
