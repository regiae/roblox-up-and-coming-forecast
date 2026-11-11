#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecast for the next 72 hours.

Method: Monte Carlo over a latent score per game. A game's score is a
power of its rank, s = -rank**GAMMA. That transform is there because rank
moves scale with depth in just the way it implies: a top game wanders
about 0.7 places a read while a deep one wanders about 6, and rank**0.35
reproduces that scaling. Each game's score drifts with its recent
trajectory (least-squares velocity over the last 30 hours, shrunk toward
zero and capped, fading with a 60-hour time constant), diffuses with
calibrated hourly noise, and sometimes surges: a rare viral jump, sized
exponentially, which is how a deep game really reaches the top (the
archive's biggest single-read jumps are 30-80 places). Averaging each
game's place over 1,500 simulated paths times 72 hours gives the
time-averaged forecast the log score rewards. Games the history cannot
name (brand-new entrants) are left to each place's unnamed share.
"""
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Parameters, tuned by backtest on the public archive (4 folds, verified
# across 3 seeds and on 8 folds). The one change from v1: the drift
# lookback is 30 hours, not 36; a shorter window tracks the chart's
# fast movers better without chasing noise.
SEED = 20261111
N_PATHS = 1500
GAMMA = 0.35
SIGMA = 0.06
DRIFT_SHRINK = 0.5
DRIFT_TAU_H = 60.0
DRIFT_LOOKBACK_H = 30.0
DRIFT_CAP = 0.06
PLAY_WEIGHT = 0.10
JUMP_RATE = 0.008
JUMP_MEAN = 1.0
SHRINK = 0.10
TARGET_MASS = 0.75
MIN_P = 0.008
MAX_ROWS = 40
FLOOR_P = 0.0015


def parse_time(text):
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_history(input_dir):
    reads = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            rows = []
            for game in doc.get("up_and_coming") or []:
                rows.append({
                    "rank": int(game["rank"]),
                    "uid": int(game["universe_id"]),
                    "playing": game.get("playing"),
                    "name": game.get("name") or "",
                })
            rows.sort(key=lambda row: row["rank"])
            reads.append({
                "at": parse_time(doc["at"]),
                "chart": rows,
                "details": doc.get("details") or {},
            })
    reads.sort(key=lambda snap: snap["at"])
    return reads


def load_task(input_dir):
    with open(os.path.join(input_dir, "task.json")) as handle:
        return json.load(handle)


def score_of(rank, playing, median_play):
    """Latent score: higher means a better place. Rank dominates; the
    playing count refines it."""
    value = -(float(rank) ** GAMMA)
    if playing and playing > 0 and median_play and median_play > 0:
        value += PLAY_WEIGHT * (math.log(playing) - math.log(median_play))
    return value


def build_games(snaps):
    """State for every game on the last chart, plus its score history."""
    last = snaps[-1]
    t_end = last["at"]
    counts = sorted(g["playing"] for g in last["chart"] if g.get("playing"))
    median_play = counts[len(counts) // 2] if counts else None
    # Latest known creation date per game (house-recorder reads carry it).
    created = {}
    for snap in snaps:
        for uid_text, det in (snap["details"] or {}).items():
            if det and det.get("created"):
                try:
                    created[int(uid_text)] = parse_time(det["created"])
                except (ValueError, TypeError):
                    pass
    # Score history per game: (hours before the end, score).
    hist = {}
    for snap in snaps:
        age_h = (t_end - snap["at"]).total_seconds() / 3600.0
        for game in snap["chart"]:
            hist.setdefault(game["uid"], []).append(
                (age_h, score_of(game["rank"], game.get("playing"), median_play)))
    games = []
    for game in last["chart"]:
        uid = game["uid"]
        points = sorted((h, s) for (h, s) in hist.get(uid, [])
                        if h <= DRIFT_LOOKBACK_H)
        drift = 0.0
        if len(points) >= 2:
            n = len(points)
            mean_h = sum(p[0] for p in points) / n
            mean_s = sum(p[1] for p in points) / n
            num = sum((p[0] - mean_h) * (p[1] - mean_s) for p in points)
            den = sum((p[0] - mean_h) ** 2 for p in points)
            if den > 0:
                # The slope is d(score)/d(hours before end); the forward
                # velocity is its negative.
                velocity = -num / den
                drift = velocity * DRIFT_SHRINK
                drift = max(-DRIFT_CAP, min(DRIFT_CAP, drift))
        exit_at = None
        if uid in created:
            # Only games published in the last 28 days are on the chart.
            exit_at = created[uid].timestamp() + 28 * 86400
        games.append({
            "uid": uid,
            "rank": game["rank"],
            "score0": score_of(game["rank"], game.get("playing"), median_play),
            "drift": drift,
            "exit_ts": exit_at,
        })
    return games, t_end


def simulate(games, t_end, hours):
    """Monte Carlo: tally[uid][place] over paths times hours."""
    rng = random.Random(SEED)
    n = len(games)
    uids = [g["uid"] for g in games]
    score0 = [g["score0"] for g in games]
    drift = [g["drift"] for g in games]
    exit_ts = [g["exit_ts"] for g in games]
    t_end_ts = t_end.timestamp()
    tally = {uid: [0] * 11 for uid in uids}  # indexed by place 1..10
    gauss = rng.gauss
    rnd = rng.random
    log = math.log
    decay = [math.exp(-h / DRIFT_TAU_H) for h in range(1, hours + 1)]
    for _ in range(N_PATHS):
        scores = list(score0)
        for h in range(hours):
            d = decay[h]
            t_now = t_end_ts + (h + 1) * 3600.0
            for i in range(n):
                if exit_ts[i] is not None and t_now >= exit_ts[i]:
                    scores[i] = -1e18
                else:
                    s = scores[i] + drift[i] * d + SIGMA * gauss(0.0, 1.0)
                    # Viral surge: rare, upward only, exponentially sized.
                    if rnd() < JUMP_RATE:
                        s += -JUMP_MEAN * log(1.0 - rnd())
                    scores[i] = s
            leaders = sorted(range(n), key=lambda i: scores[i], reverse=True)[:10]
            for place, i in enumerate(leaders, start=1):
                tally[uids[i]][place] += 1
    return tally


def build_forecast(tally, games, places, total):
    # Persistence prior: the starter's forecast (the chart does not move).
    last_order = sorted(games, key=lambda g: g["rank"])
    prior = {}
    for k in range(1, places + 1):
        prior[k] = {}
        if k <= len(last_order):
            prior[k][last_order[k - 1]["uid"]] = 0.5
        if k == 1 and len(last_order) > 1:
            prior[k][last_order[1]["uid"]] = 0.3
        elif k > 1:
            if k - 2 >= 0:
                prior[k][last_order[k - 2]["uid"]] = 0.15
            if k < len(last_order):
                prior[k][last_order[k]["uid"]] = 0.15
    probabilities = []
    exp_rank = {}
    for game in games:
        c = tally[game["uid"]]
        in_top = sum(c[1:])
        # Expected rank, counting rank 11 while outside the top 10.
        exp_rank[game["uid"]] = (sum(p * c[p] for p in range(1, places + 1))
                                 + (places + 1) * (total - in_top)) / total
    top10 = [g["uid"] for g in sorted(games, key=lambda g: exp_rank[g["uid"]])[:places]]
    for k in range(1, places + 1):
        raw = {}
        for game in games:
            p = tally[game["uid"]][k] / total
            if p > 0:
                raw[game["uid"]] = p
        # Calibrate: shrink the Monte Carlo probabilities toward the prior.
        mixed = {}
        for uid, p in raw.items():
            mixed[uid] = (1.0 - SHRINK) * p + SHRINK * prior[k].get(uid, 0.0)
        for uid, p in prior[k].items():
            if uid not in mixed:
                mixed[uid] = SHRINK * p
        # Keep only the strongest candidates, floor them, scale to target.
        kept = sorted(mixed.items(), key=lambda kv: kv[1], reverse=True)[:MAX_ROWS]
        kept = [(uid, max(p, FLOOR_P)) for uid, p in kept if p >= MIN_P]
        if not kept:
            continue
        s = sum(p for _, p in kept)
        scale = TARGET_MASS / s if s > TARGET_MASS else 1.0
        for uid, p in kept:
            probabilities.append({"universe_id": uid, "place": k,
                                  "p": round(p * scale, 6)})
    return {"probabilities": probabilities, "top10": top10}


def main():
    task = load_task(INPUT_DIR)
    places = int(task.get("places", 10))
    # The scored window comes from task.json, never hard-coded.
    window = task.get("window") or {}
    slots = int(window.get("slots", 72))
    slot_min = int(window.get("slot_minutes", 60))
    hours = max(1, slots * slot_min // 60)
    snaps = load_history(INPUT_DIR)
    games, t_end = build_games(snaps)
    tally = simulate(games, t_end, hours)
    forecast = build_forecast(tally, games, places, N_PATHS * hours)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out, "w") as handle:
        handle.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
