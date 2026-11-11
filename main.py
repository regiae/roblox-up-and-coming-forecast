#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster.

Monte Carlo over latent scores, calibrated on the chart's own history. Each game
holds a latent score whose order is the chart's order. The score is a power of the
rank, s = -rank**GAMMA, because the chart's ranks diffuse the way they do: a top
game moves ~0.7 places a read, a deep one ~6, and rank**0.35 reproduces that
scaling (the noise that moves a rank-2 game 0.7 places moves a rank-60 game ~6).
The score drifts with the game's recent trajectory (momentum, shrunk and fading),
diffuses with calibrated noise, and occasionally surges: a viral jump, sized
exponentially, which is how deep games really reach the top (the history's
biggest jumps are 30-80 places in one read, far past any diffusion). Simulating
72 hours forward and averaging the place each game holds gives an honest,
time-averaged forecast for the whole scored window, which is what the log score
rewards. Games the model cannot name (new entrants) are left to the unnamed
share of each place.
"""
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunable parameters (calibrated on the public archive, tuned by backtest) ---
SEED = 20261111
N_PATHS = 1500          # Monte Carlo paths
HOURS = 72              # scored window length
GAMMA = 0.35            # score transform: s = -rank**GAMMA
SIGMA = 0.06            # per-hour latent-score noise (std), in score units
DRIFT_SHRINK = 0.5      # shrink the momentum drift toward 0
DRIFT_TAU_H = 60.0      # momentum decay time constant, hours
DRIFT_LOOKBACK_H = 36.0  # window used to estimate a game's score velocity
DRIFT_CAP = 0.06        # largest |drift| per hour, in score units
PLAY_WEIGHT = 0.10      # weight of log(playing) in the latent score
JUMP_RATE = 0.008       # per hour, per game: probability of a viral surge
JUMP_MEAN = 1.0         # mean size of a surge, in score units (exponential)
SHRINK = 0.10           # mix of the Monte Carlo probs toward the persistence prior
TARGET_MASS = 0.75      # total probability named per place (rest left unnamed)
MIN_P = 0.008           # smallest p a named game gets at a place
MAX_ROWS = 40           # most games named at one place
FLOOR_P = 0.0015        # floor for a named game's p after calibration


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chart = []
            for g in d.get("up_and_coming") or []:
                chart.append({
                    "rank": int(g["rank"]),
                    "uid": int(g["universe_id"]),
                    "playing": g.get("playing"),
                    "name": g.get("name") or "",
                })
            chart.sort(key=lambda g: g["rank"])
            details = d.get("details") or {}
            snaps.append({"at": parse_time(d["at"]), "chart": chart, "details": details})
    snaps.sort(key=lambda s: s["at"])
    return snaps


def load_task(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        return json.load(f)


def score_of(rank, playing, play_med):
    """Latent score: higher is a better place. Rank term dominates; playing refines."""
    base = -(float(rank) ** GAMMA)
    if playing and playing > 0 and play_med and play_med > 0:
        base += PLAY_WEIGHT * (math.log(playing) - math.log(play_med))
    return base


def build_games(snaps):
    """State for every game on the last chart, plus its score trajectory."""
    last = snaps[-1]
    t_end = last["at"]
    plays = sorted(g["playing"] for g in last["chart"] if g.get("playing"))
    play_med = plays[len(plays) // 2] if plays else None
    # created dates: latest details entry per game (house recorder reads carry them)
    created = {}
    for s in snaps:
        for uid_s, det in (s["details"] or {}).items():
            if det and det.get("created"):
                try:
                    created[int(uid_s)] = parse_time(det["created"])
                except (ValueError, TypeError):
                    pass
    # score trajectory per game: list of (hours before end, score)
    traj = {}
    for s in snaps:
        dt_h = (t_end - s["at"]).total_seconds() / 3600.0
        for g in s["chart"]:
            traj.setdefault(g["uid"], []).append(
                (dt_h, score_of(g["rank"], g.get("playing"), play_med)))
    games = []
    for g in last["chart"]:
        uid = g["uid"]
        # score velocity over the lookback window (least squares over the points)
        pts = sorted((h, sc) for (h, sc) in traj.get(uid, []) if h <= DRIFT_LOOKBACK_H)
        drift = 0.0
        if len(pts) >= 2:
            n = len(pts)
            mh = sum(p[0] for p in pts) / n
            ms = sum(p[1] for p in pts) / n
            num = sum((p[0] - mh) * (p[1] - ms) for p in pts)
            den = sum((p[0] - mh) ** 2 for p in pts)
            if den > 0:
                # slope is d(score)/d(hours before end); velocity forward is its negative
                vel = -num / den
                drift = vel * DRIFT_SHRINK
                drift = max(-DRIFT_CAP, min(DRIFT_CAP, drift))
        exit_at = None
        if uid in created:
            # the chart lists games published in the last 28 days
            exit_at = created[uid].timestamp() + 28 * 86400
        games.append({
            "uid": uid,
            "rank": g["rank"],
            "score0": score_of(g["rank"], g.get("playing"), play_med),
            "drift": drift,
            "exit_ts": exit_at,
        })
    return games, t_end


def simulate(games, t_end):
    """Monte Carlo: counts[uid][place] over paths x hours. Returns counts."""
    rng = random.Random(SEED)
    n = len(games)
    uids = [g["uid"] for g in games]
    score0 = [g["score0"] for g in games]
    drift = [g["drift"] for g in games]
    exit_ts = [g["exit_ts"] for g in games]
    t_end_ts = t_end.timestamp()
    counts = {uid: [0] * 11 for uid in uids}  # index by place 1..10
    gauss = rng.gauss
    rnd = rng.random
    log = math.log
    decay = [math.exp(-h / DRIFT_TAU_H) for h in range(1, HOURS + 1)]
    for _ in range(N_PATHS):
        scores = list(score0)
        for h in range(HOURS):
            d = decay[h]
            t_now = t_end_ts + (h + 1) * 3600.0
            for i in range(n):
                if exit_ts[i] is not None and t_now >= exit_ts[i]:
                    scores[i] = -1e18
                else:
                    s = scores[i] + drift[i] * d + SIGMA * gauss(0.0, 1.0)
                    # viral surge: rare, upward only, exponentially sized
                    if rnd() < JUMP_RATE:
                        s += -JUMP_MEAN * log(1.0 - rnd())
                    scores[i] = s
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:10]
            for place, i in enumerate(order, start=1):
                counts[uids[i]][place] += 1
    return counts


def build_forecast(counts, games, places):
    total = N_PATHS * HOURS
    # persistence prior: the starter's forecast (chart does not move)
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
    for g in games:
        c = counts[g["uid"]]
        in_top = sum(c[1:])
        # expected rank: rank 11 when outside the top 10
        exp_rank[g["uid"]] = (sum(p * c[p] for p in range(1, places + 1))
                              + (places + 1) * (total - in_top)) / total
    top10 = [g["uid"] for g in sorted(games, key=lambda g: exp_rank[g["uid"]])[:places]]
    for k in range(1, places + 1):
        raw = {}
        for g in games:
            p = counts[g["uid"]][k] / total
            if p > 0:
                raw[g["uid"]] = p
        # calibrate: shrink toward the persistence prior
        cal = {}
        for uid, p in raw.items():
            cal[uid] = (1.0 - SHRINK) * p + SHRINK * prior[k].get(uid, 0.0)
        for uid, p in prior[k].items():
            if uid not in cal:
                cal[uid] = SHRINK * p
        # keep only the strongest candidates, floor, and scale to TARGET_MASS
        items = sorted(cal.items(), key=lambda kv: kv[1], reverse=True)[:MAX_ROWS]
        items = [(uid, max(p, FLOOR_P)) for uid, p in items if p >= MIN_P]
        if not items:
            continue
        s = sum(p for _, p in items)
        scale = TARGET_MASS / s if s > TARGET_MASS else 1.0
        for uid, p in items:
            probabilities.append({"universe_id": uid, "place": k, "p": round(p * scale, 6)})
    return {"probabilities": probabilities, "top10": top10}


def main():
    task = load_task(INPUT_DIR)
    places = int(task.get("places", 10))
    # read the scored window from task.json rather than assuming it
    global HOURS
    win = task.get("window") or {}
    slots = int(win.get("slots", 72))
    slot_min = int(win.get("slot_minutes", 60))
    HOURS = max(1, slots * slot_min // 60)
    snaps = load_history(INPUT_DIR)
    games, t_end = build_games(snaps)
    counts = simulate(games, t_end)
    forecast = build_forecast(counts, games, places)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out, "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
