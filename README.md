# Extreme 3-VWAP Mean Reversion — Backtest Results

## What this is

A backtest of the `Extreme 3-VWAP Mean Reversion` Pine Script strategy over
5 years of 5-minute data (2019-2024), on two futures contracts: **ES**
(S&P 500) and **NQ** (NASDAQ 100).

## Results

| | ES (S&P 500) | NQ (NASDAQ 100) |
|---|---|---|
| Total bars | 353,206 | 329,458 |
| Signals fired | 29 (short only) | 32 (short only) |
| Trades | 22 | 22 |
| Win rate | 63.6% | 59.1% |
| Total return (5 years) | 1.82% | 1.06% |
| Avg trade | +0.083% | +0.049% |

## The core finding: why so few signals

The strategy requires **four things at once**:
1. Price to have diverged sharply from all 3 VWAPs (session, 12, 78)
2. ADX to be low (i.e. a calm, sideways market)
3. Volume to be unusually high (a spike)
4. (Long signals are disabled by design — `allowLong = false`)

The problem: when price runs far away with strong volume, that **usually
means a trend is forming, not calm conditions** — so ADX tends to rise
exactly when you need it to be low. The two conditions tend to cancel
each other out. Result: across 5+ years of data, only ~6 signals per year.

## Why I did this — and what I want to show

**I am not recommending you trade this strategy.** The point here was
educational: to show, hands-on, one of the most common mistakes people
make when evaluating a trading system — **being impressed by a good win
rate without checking how many trades it's actually based on.**

A 60-64% win rate sounds great. But when it comes from just **22 trades
across 5 full years**, it barely means anything. With such a small
sample:

- It could easily be luck. Flip a coin 22 times — a 13-9 or even 14-8
  split wouldn't surprise anyone.
- The confidence interval around a 60% win rate with n=22 is huge
  (realistically somewhere around 40%-80%). You genuinely don't know
  whether the strategy has any real edge.
- Both instruments (ES, NQ) produced almost the same number of trades
  and similarly marginal, barely-positive results — that pattern is
  more consistent with "noise around zero" than with a real, repeatable
  edge.
- The actual net gain after 5 years is only ~1-2% — less than you'd
  have earned just parking the money in a plain interest-bearing
  account, with zero risk.

**The takeaway I'm keeping from this:** any single statistic (win rate,
Sharpe ratio, whatever) means nothing without its sample size attached.
Before trusting any backtest, ask first "how many trades support this,
and across how large/varied a sample?" — not "how good does the win
rate look?"

## Where the script is

`vwap_mean_reversion_backtest.py` — reproduces the Pine Script logic in
Python/pandas, auto-downloads the Kaggle dataset via `kagglehub`, and
runs the backtest on either ES or NQ (switch via `TICKER_HINT`).
