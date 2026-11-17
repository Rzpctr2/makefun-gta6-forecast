"""gta6-twitch-ccu-forecast-01 entry: console-launch ratio curve with a
pre-midnight anchor (Python 3.13, standard library only).

Method (chosen from local tests on the 4 practice cases; see README):
  - For each hour h = 2..24, take every past big launch in the case's
    `comparables` that is console-only (its `platforms` has no PC — the
    final game, GTA VI, is console-only, as are the practice launches),
    and compute log(viewers(h) / anchor) on its release day, where
      anchor = p1^0.15 * pre^0.85,
    p1 is that launch's hour-1 print and pre is the geometric mean of its
    hourly viewers at 21:00/22:00/23:00 UTC the day before release.
  - The forecast median for hour h is the target's anchor times the
    MEDIAN of those log-ratios. The target's anchor is built the same way
    from its own hour-1 print and its pre-midnight hourly history.
  - Quantiles are lognormal around that median with
    sigma_h = 0.12 * sqrt(h - 1), using the standard normal quantiles at
    0.05/0.25/0.5/0.75/0.95.

Why this shape: hour 1 of a rolling-midnight launch is a depressed
anchor (pre-release hype streams wind down at midnight while the US is
still asleep), so the pre-midnight level carries most of the scale
information; the console-only median curve captures the day's shape
(dip to ~0.7x in the small hours, US unlocks from hour 5, evening peak
~2x the anchor). Fallbacks keep the output valid when data is thin:
fewer than 6 console-only ratios for an hour -> all comparables; still
none -> persistence (ratio 1); missing pre-midnight history -> anchor
is the hour-1 print alone.

Local practice result (ground truth = SullyGnome "Average viewers" for
the 4 practice release days, hour-1 values verified equal to the cases'
`known` prints): mean pinball loss 0.183 vs house baseline 0.387,
per case p00 0.222 / p01 0.145 / p02 0.180 / p03 0.184.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim,
standard library only, no network. It reads every <id>.in in the input
directory and writes <id>.out to the output directory:
    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}
exactly 23 rows, hours 2..24, each q a finite number above 0 and at most
1e9, non-decreasing. Deterministic: no randomness, no clock, no network.
Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
from datetime import date, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Anchor = p1^A * pre^(1-A): weight on the hour-1 print, the rest on the
# geometric mean of the pre-midnight hourly viewers (21/22/23 UTC).
A = 0.15
# Lognormal spread of an hour's forecast: sigma_h = SPREAD * sqrt(h - 1).
SPREAD = 0.12
# Pre-midnight hours (UTC, day before release) feeding the anchor.
PRE_HOURS = ("21", "22", "23")
# Minimum log-ratios for the console-only curve before falling back.
MIN_N = 6


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the starter rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _day_hours(points, day: str) -> dict:
    """Hourly viewers {hour 1..24: viewers} for one UTC day of a series."""
    out = {}
    for p in points or []:
        f = p.get("from")
        if not f or not f.startswith(day):
            continue
        v = p.get("viewers")
        if v is None or v <= 0:
            continue
        h = int(f[11:13]) + 1
        if 1 <= h <= 24:
            out[h] = float(v)
    return out


def _pre_level(points, day_start: str):
    """Geometric mean of viewers at 21/22/23 UTC the day before day_start.

    Returns None when the series has none of those hours (the anchor then
    falls back to the hour-1 print alone).
    """
    if not day_start:
        return None
    prev_day = str(date.fromisoformat(day_start[:10]) - timedelta(days=1))
    vals = []
    for p in points or []:
        f = p.get("from")
        if not f or not f.startswith(prev_day):
            continue
        if f[11:13] not in PRE_HOURS:
            continue
        v = p.get("viewers")
        if v is not None and v > 0:
            vals.append(float(v))
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _anchor(p1: float, pre) -> float:
    """Weighted geometric mean of the hour-1 print and the pre-midnight level."""
    if pre is None or pre <= 0:
        return p1
    return math.exp(A * math.log(p1) + (1.0 - A) * math.log(pre))


def _comparable_curves(case: dict):
    """Precompute each comparable's release-day hours and its anchor.

    A comparable only counts when its release day has an hour-1 point;
    the GTA VI category's own pre-release series has none on its
    day_start, so it never enters the curve.
    """
    curves = []
    for c in case.get("comparables") or []:
        day = (c.get("day_start") or "")[:10]
        if not day:
            continue
        hourly = c.get("hourly") or []
        byhour = _day_hours(hourly, day)
        if 1 not in byhour:
            continue
        pre = _pre_level(hourly, c.get("day_start") or "")
        curves.append((byhour, _anchor(byhour[1], pre),
                       "PC" not in (c.get("platforms") or "")))
    return curves


def _log_ratios(curves, hour: int, console_only: bool):
    out = []
    for byhour, anch, is_console in curves:
        if console_only and not is_console:
            continue
        if hour not in byhour or anch <= 0:
            continue
        out.append(math.log(byhour[hour] / anch))
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = {}
    for k in case.get("known") or []:
        if isinstance(k, dict) and k.get("hour") is not None:
            known[k["hour"]] = k
    p1 = None
    if 1 in known and known[1].get("viewers"):
        p1 = float(known[1]["viewers"])
    if p1 is None or p1 <= 0:
        # No hour-1 print: fall back to the last pre-checkpoint hourly point.
        hh = case.get("history_hourly") or []
        p1 = float(hh[-1]["viewers"]) if hh and hh[-1].get("viewers") else 1.0
    p1 = max(1.0, p1)
    pre = _pre_level(case.get("history_hourly") or [], case.get("day_start") or "")
    anch = _anchor(p1, pre)
    curves = _comparable_curves(case)
    hours = []
    for h in range(2, 25):
        lrs = _log_ratios(curves, h, console_only=True)
        if len(lrs) < MIN_N:
            lrs = _log_ratios(curves, h, console_only=False)
        med = statistics.median(lrs) if lrs else 0.0
        sigma = SPREAD * math.sqrt(h - 1)
        med_v = anch * math.exp(med)
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(med_v * math.exp(z * sigma))
            q = min(max(q, 1.0), 1e9)
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
