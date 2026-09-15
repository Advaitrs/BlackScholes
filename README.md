# Black-Scholes

The Black-Scholes model prices European options under the assumption that the underlying follows geometric Brownian motion with constant volatility and interest rates. This project implements the closed-form call and put formulas, along with the analytic Greeks (delta, gamma, vega, theta, rho), in clean Python.

## Layout

```
black-scholes/
├── main.py
├── requirements.txt
├── src/
│   └── black_scholes.py
├── tests/
│   └── test_black_scholes.py
└── README.md
```

## Class design

`BlackScholes` holds a single market snapshot — spot, strike, rate, volatility, and time to expiry — so prices and Greeks are always computed from the same inputs.

Greeks are separate methods rather than one bulk return: each has its own closed-form expression, and none of them mutate the stored parameters. Methods that don't change state read like pure math (`call_price()`, `delta(is_call=True)`, and so on).

Approximations that use a different algorithm (for example American options via a binomial tree) belong in their own class, not bolted onto this one.

## Setup

```bash
python -m pip install -r requirements.txt
```

```bash
python main.py
python -m pytest -q
```

## Backtesting

The plan is to pull live or historical option quotes from a public market-data API and compare them against model prices and implied volatilities — so the pricer can be checked against the market, not only against textbook numbers.
