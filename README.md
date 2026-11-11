# Roblox Up-and-Coming top-10 forecaster

Monte Carlo over latent scores. Each game holds a latent score that is a
power of its rank, `s = -rank**0.35`, because rank moves scale with depth
in just the way that transform implies: a top game wanders about 0.7
places per read while a deep one wanders about 6. The score drifts with
the game's recent trajectory (least-squares velocity over the last 30
hours, shrunk and capped, fading with a 60-hour time constant), diffuses
with calibrated hourly noise, and occasionally surges: a rare viral jump,
exponentially sized, which is how a deep game really reaches the top.

Simulating 72 hours forward over 1,500 paths and averaging the place
each game holds gives an honest, time-averaged forecast for the whole
scored window, which is what the log score rewards. Games the model
cannot name (new entrants) are left to the unnamed share of each place.

## Parameters

Tuned by backtest on the public archive: 4 folds, verified across 3
seeds and on 8 folds. The drift lookback is 30 hours (v1 used 36); a
shorter window tracks the chart's fast movers better without chasing
noise. Everything else matches the v1 calibration: rank power 0.35,
hourly noise 0.06, drift shrink 0.5 with a 60-hour decay, drift cap
0.06, playing weight 0.10, jump rate 0.008/hour with mean size 1.0,
0.10 shrink toward the persistence prior, 0.75 of each place named.

## Backtest (log score, higher is better)

| config | 4-fold mean | 8-fold mean | practice fold |
|--------|-------------|-------------|---------------|
| v1 (lookback 36h) | -35.995 | -37.233 | -33.120 |
| v2 (lookback 30h) | -35.806 | -37.075 | -33.181 |

The v2 gain (+0.19 on 4 folds, +0.16 on 8 folds) replicates across
seeds 20261111, 1, and 42; the practice fold is unchanged within
noise. The house baseline scores -43.512 on the practice fold.

## Run

```
python3 main.py
```

Reads the chart history from `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes `forecast.json` to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Standard library only, no network, deterministic (fixed
seed), about 7 seconds.
