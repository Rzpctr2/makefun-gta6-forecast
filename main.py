"""gta6-twitch-ccu-forecast-01 entry, v3: ensemble median + pre-midnight
ramp correction + flat honest spread (Python 3.13, standard library only).

Method (chosen from local tests on the 4 practice cases, a leave-one-out
run over the 38 kit comparables, and a simulated-final pool; see README-v3):
  - Anchor: the pre-midnight level only -- the geometric mean of the
    target's hourly viewers at 21:00/22:00/23:00 UTC the day before release.
    Hour 1 of a rolling-midnight launch is a depressed print (pre-release
    hype streams wind down at midnight while the US is still asleep), and
    mixing it in only adds noise. When the pre-midnight level is missing,
    the anchor falls back to the hour-1 print alone.
  - Median curve: an ensemble of two scale-aware curves, averaged in log
    space with weights 0.3 / 0.7:
      (a) the v2 scale-weighted median: for each hour h, the median of
          log(viewers(h) / anchor) over console-only, tier-"comparable"
          comparables, each weighted by a Gaussian kernel on the
          log-distance between its anchor and the target's anchor
          (bandwidth 1.0), with the fallback chain clean -> console-only
          (any tier) -> all comparables of tier "comparable", each step
          taken only when the previous pool has fewer than 6 ratios;
      (b) a per-hour log-log regression: log(viewers(h)) regressed on
          log(anchor) over the console-only, tier-"comparable" pool
          (ordinary least squares, one slope and intercept per hour),
          evaluated at the target's anchor. Slopes run 0.68-0.98, i.e.
          launch-day shape scales sublinearly with size: bigger launches
          peak relatively lower. As insurance the regression median is
          clamped to [0.25x, 4x] of the (a) median; the clamp never binds
          on any tested case (max observed ratio 1.65x).
    The two components disagree most where each is weakest -- (a) is noisy
    hour to hour on a 7-curve pool, (b) extrapolates a line -- and their
    average beats either alone on practice, on the simulated final, and
    on leave-one-out over the big comparables.
  - Pre-midnight ramp correction: g = log(viewers at 23:00 / viewers at
    18:00 UTC the day before release) measures how fast the unlock wave
    is already building at midnight. Across the big comparables, the
    residual of the ensemble median correlates with g (r ~ +0.6 in the
    first hours): a steep ramp means the early UTC hours stay higher than
    the average curve. The median is multiplied by exp(b_h * dg), where
    dg = g - (-0.099) (the pool's median g), clamped to [-1.0, +1.5],
    with b_h = 0.35 for hours 2-5, 0.20 for hours 6-12, 0 afterwards.
    When the 18:00/23:00 points are missing the correction is skipped.
  - Quantiles are lognormal around the median with a CONSTANT sigma of
    0.35 for every hour. The residual spread of the median is roughly
    flat across hours (~0.2 on practice launches, ~0.5 on the diverse
    big-launch pool), so the v2 spread 0.11*sqrt(h-1) is far too narrow
    in the small hours and no better late; a flat 0.35 wins on practice,
    on the simulated final, and on leave-one-out alike.

Deliberately NOT done: the practice leaderboard is topped by entries
that embed/memorize the practice answers with a ~0 spread (practice loss
0.000001). That scores nothing on the final -- the prize is decided by
the one real GTA VI case, where a memorized curve with a near-zero spread
scores its full log error -- so no practice answers are embedded here
and the spread stays honest.

Local results (ground truth = SullyGnome "Average viewers" for the 4
practice release days, hour-1 values verified equal to the cases'
`known` prints; pinball loss exactly as RULES.txt defines it):
  practice mean 0.141 (p00 0.175 / p01 0.130 / p02 0.110 / p03 0.149)
    vs v2 0.156 and house baseline 0.387.
  simulated final (practice launches added as comparables, target left
    out, 41 total): 0.142 vs v2 0.161.
  leave-one-out over the 21 big comparables (anchor >= 50k): 0.260
    vs v2 0.276; over the 7 console-only tier-"comparable" ones: 0.230
    vs v2 0.326.

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
import re
import statistics
from datetime import date, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Lognormal spread of every hour's forecast (flat, see docstring).
SIGMA = 0.35
# Pre-midnight hours (UTC, day before release) feeding the anchor.
PRE_HOURS = ("21", "22", "23")
# Minimum log-ratios in a pool before falling back to a wider pool.
MIN_N = 6
# Gaussian kernel bandwidth (in log-anchor units) for the scale weights.
SCALE_BW = 1.0
# Ensemble weight on the scale-weighted-median component (rest: log-log).
W_V2 = 0.3
# Pre-midnight ramp correction: pool median g, clamp on dg, slopes by hour.
G_MED = -0.099
G_CLAMP = (-1.0, 1.5)
# Insurance clamp on the log-log median relative to the v2 median.
LOGLOG_CLAMP = (0.25, 4.0)


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


def _prev_day_hours(points, day_start: str) -> dict:
    """Hourly viewers {clock hour: viewers} for the day before day_start."""
    if not day_start:
        return {}
    prev_day = str(date.fromisoformat(day_start[:10]) - timedelta(days=1))
    out = {}
    for p in points or []:
        f = p.get("from")
        if not f or not f.startswith(prev_day):
            continue
        v = p.get("viewers")
        if v is not None and v > 0:
            out[int(f[11:13])] = float(v)
    return out


def _pre_level(points, day_start: str):
    """Geometric mean of viewers at 21/22/23 UTC the day before day_start.

    Returns None when the series has none of those hours (the anchor then
    falls back to the hour-1 print alone).
    """
    prev = _prev_day_hours(points, day_start)
    vals = [prev[int(h)] for h in PRE_HOURS if int(h) in prev]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _pre_growth(points, day_start: str):
    """log(viewers 23:00 / viewers 18:00 UTC) the day before day_start.

    Returns None when either hour is missing (the ramp correction is
    then skipped).
    """
    prev = _prev_day_hours(points, day_start)
    if 23 in prev and 18 in prev and prev[18] > 0:
        return math.log(prev[23] / prev[18])
    return None


def _anchor(p1: float, pre) -> float:
    """The anchor is the pre-midnight level alone (hour 1 adds noise)."""
    if pre is None or pre <= 0:
        return p1
    return pre


def _tier(comp: dict) -> str:
    """The comparable's tier from its unlock note ('Tier: <tier>')."""
    m = re.search(r"Tier:\s*([a-z\-]+)", comp.get("unlock") or "")
    return m.group(1) if m else "unknown"


def _comparable_curves(case: dict):
    """Precompute each comparable's release-day hours, anchor and tier.

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
                       "PC" not in (c.get("platforms") or ""), _tier(c)))
    return curves


def _weighted_median(pairs):
    """Median of (value, weight) pairs, weighted by the second element."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for x, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return x
    return pairs[-1][0]


def _log_ratios(curves, hour: int, anchor: float, console_only: bool,
                tier: str | None):
    """Scale-weighted (log-ratio, weight) pairs for one hour.

    tier narrows the pool ("comparable" keeps only that tier); None keeps
    every tier. Weights are a Gaussian kernel on the log-anchor distance
    to the target's anchor, so similarly-sized launches count most.
    """
    out = []
    log_anchor = math.log(anchor)
    for byhour, anch, is_console, t in curves:
        if console_only and not is_console:
            continue
        if tier is not None and t != tier:
            continue
        if hour not in byhour or anch <= 0:
            continue
        d = (math.log(anch) - log_anchor) / SCALE_BW
        out.append((math.log(byhour[hour] / anch), math.exp(-0.5 * d * d)))
    return out


def _v2_median(curves, hour: int, anchor: float) -> float:
    """Scale-weighted median log-ratio curve (the v2 component)."""
    pairs = _log_ratios(curves, hour, anchor, console_only=True, tier="comparable")
    if len(pairs) < MIN_N:
        pairs = _log_ratios(curves, hour, anchor, console_only=True, tier=None)
    if len(pairs) < MIN_N:
        pairs = _log_ratios(curves, hour, anchor, console_only=False, tier="comparable")
    med = _weighted_median(pairs) if pairs else 0.0
    return anchor * math.exp(med)


def _ols(xs, ys):
    """Least-squares line y ~ a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _loglog_median(curves, hour: int, anchor: float, v2_med: float) -> float:
    """Per-hour log-log regression median over the clean pool.

    Falls back to the pooled median log-ratio when too few points, and
    is clamped to [0.25x, 4x] of the v2 median as insurance against a
    wild extrapolation at an anchor far outside the pool's range.
    """
    xs, ys = [], []
    for byhour, anch, is_console, t in curves:
        if not is_console or t != "comparable":
            continue
        if hour in byhour and anch > 0:
            xs.append(math.log(anch))
            ys.append(math.log(byhour[hour]))
    if len(xs) >= 4:
        a, b = _ols(xs, ys)
        med = math.exp(a + b * math.log(anchor))
    else:
        rs = [math.log(byhour[hour] / anch) for byhour, anch, _, _
              in curves if hour in byhour and anch > 0]
        med = anchor * math.exp(statistics.median(rs)) if rs else anchor
    lo = LOGLOG_CLAMP[0] * v2_med
    hi = LOGLOG_CLAMP[1] * v2_med
    return min(max(med, lo), hi)


def _g_slope(hour: int) -> float:
    """Ramp-correction slope for one hour (0 outside hours 2-12)."""
    if hour <= 5:
        return 0.35
    if hour <= 12:
        return 0.20
    return 0.0


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
    history = case.get("history_hourly") or []
    pre = _pre_level(history, case.get("day_start") or "")
    anch = _anchor(p1, pre)
    curves = _comparable_curves(case)
    # Pre-midnight ramp correction, shared across hours.
    g = _pre_growth(history, case.get("day_start") or "")
    dg = None
    if g is not None:
        dg = min(max(g - G_MED, G_CLAMP[0]), G_CLAMP[1])
    hours = []
    for h in range(2, 25):
        v2_med = _v2_median(curves, h, anch)
        ll_med = _loglog_median(curves, h, anch, v2_med)
        med_v = math.exp(W_V2 * math.log(v2_med) + (1.0 - W_V2) * math.log(ll_med))
        if dg is not None:
            b = _g_slope(h)
            if b:
                med_v *= math.exp(b * dg)
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(med_v * math.exp(z * SIGMA))
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
