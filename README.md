# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/regiae/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/regiae/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A Monte Carlo forecaster for Roblox's Up-and-Coming chart: each game holds a latent score whose order is the chart's order, the score drifts with the game's recent momentum, diffuses with noise calibrated on the chart's own history, and occasionally surges the way viral games really do. Simulating the 72 hours after entries close and averaging where each game lands gives an honest, time-averaged top-10 forecast, scored by log score.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads `/input/history.ndjson` and `/input/task.json` (or the `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` overrides) and writes `forecast.json`: for each of the top 10 places, a probability for each game named at that place, plus the guessed top 10 in order. Standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wcx9zzh3&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
