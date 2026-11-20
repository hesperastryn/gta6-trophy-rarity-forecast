"""Forecast GTA VI's PS5 trophy earn rates on Exophase one week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13,
standard library only, no network). The grader runs this file once per case:
it reads every <id>.in in the input directory and writes <id>.out to the
output directory, one row per trophy holding five quantiles (the case's
levels) of that trophy's earn rate, in percent, at the case's target.

The model blends three forecasts of each trophy's rate, all fitted by
time-weighted least squares on the clock x = ln(1 + hours since release)
over the last 24 h of reads (each read weighs its gap to the next, decayed
with a 12 h scale; a stale head the page repeated verbatim before a jump in
players tracked is dropped). The last day is what counts: day-one reads
describe a crowd that has already been replaced by later, slower players.

1. A saturating curve: rate = P / (1 + exp(-(a + b*x))) with the cap P chosen
   from a grid by fit quality. Earn rates bend as the crowd reaches a
   trophy: chapter completions pile up fast, then flatten near the share of
   players who will ever get there in week one. A straight line on any scale
   overshoots that bend; the cap is what bends.
2. A decomposition: earners and players tracked are extrapolated separately
   (log earners and log players, slopes shrunk toward their group means and
   damped) and divided. This is the piece that sees dilution, when a
   trophy's earners stall while the tracked crowd keeps growing.
3. A direct line: the rate's logit on the same clock, slope shrunk and
   damped. It anchors the blend when the other two disagree wildly.

The blend is 50/30/20 on the logit scale. The spread is the quadrature of a
random-walk term (0.35 logit units per root day of horizon) and a
fit-uncertainty term from the components' residuals, floored at 0.40 and
capped at 2.00; a trophy with no earners yet gets at least 1.10, and the one
trophy every tracked player has is centred on 99.99 (the score's own clamp)
with a 0.45 spread. On the kit's
practice cases (one stale page repeated) every fit is flat, so the program
degrades to persistence with an honest spread.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

WINDOW_HOURS = 24.0        # fits use at most this much of the history's tail
MIN_READS_KEPT = 14        # ...but never fewer than this many reads
DECAY_HOURS = 12.0         # recency scale for read weights
TAU_SLOPE = 0.9            # prior sd of a slope around its group mean
DAMP_EARNERS = 0.85        # extrapolation damping, earner/players slopes
DAMP_PLAYERS = 0.80
DAMP_DIRECT = 0.85         # extrapolation damping, direct logit slope
DAMP_CAP = 0.90            # extrapolation damping, saturating-curve slope
W_CAP = 0.50               # blend weights on the logit scale
W_DEC = 0.30
W_DIR = 0.20
WALK_PER_ROOT_DAY = 0.35   # random-walk spread, logit units per sqrt(day)
SIGMA_FLOOR = 0.40
SIGMA_CAP = 2.00
ZERO_EARNER_SIGMA = 1.10   # a trophy with no earners yet is this uncertain
FULL_RATE_SIGMA = 0.45     # a trophy at 100% stays near 100%
PLAYERS_GROWTH_CAP = 6.0   # players tracked at most this many times the last
CAP_GRID = 28              # saturating-curve caps tried per trophy


def parse_time(text):
    """ISO 8601 text -> aware datetime (the case files use a trailing Z)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    top = math.exp(value)
    return 100.0 * top / (1.0 + top)


def norm_ppf(prob):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    low, high = 0.02425, 1 - 0.02425
    if prob < low:
        q = math.sqrt(-2 * math.log(prob))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if prob > high:
        q = math.sqrt(-2 * math.log(1 - prob))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = prob - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = value * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000
    return int(result) if float(result).is_integer() else result


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (intercept, slope, residual_sd, slope_se); slope is None when x
    has no spread, and the whole fit is None when the weights are empty.
    """
    total = sum(ws)
    if total <= 0:
        return None
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (mean_y, None, 0.0, None)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sum(1 for w in ws if w > 0) - 2)
    resid_sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / total * len(ws) / dof)
    slope_se = resid_sd / math.sqrt(sxx)
    return (intercept, slope, resid_sd, slope_se)


def refit_intercept(xs, ys, ws, slope):
    total = sum(ws)
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total
    return mean_y - slope * mean_x


def shrink_slope(slope, slope_se, group_mean):
    """Pull a noisy slope toward its group's mean slope (empirical Bayes)."""
    if slope is None:
        return group_mean
    if slope_se is None or slope_se <= 0:
        return slope
    weight = TAU_SLOPE ** 2 / (TAU_SLOPE ** 2 + slope_se ** 2)
    return group_mean + weight * (slope - group_mean)


def prepare_reads(case, anchor):
    """History -> (hours since anchor, players, per-read rates, weights).

    Keeps the last WINDOW_HOURS of reads (at least MIN_READS_KEPT), drops a
    stale head the page repeated verbatim before a jump in players tracked,
    and weights each read by its gap to the next, decayed toward the past.
    """
    history = [h for h in case["history"] if h.get("players_tracked")]
    times = [(parse_time(h["at"]) - anchor).total_seconds() / 3600.0 for h in history]
    checkpoint = (parse_time(case["at"]) - anchor).total_seconds() / 3600.0
    start = 0
    while len(history) - start > MIN_READS_KEPT and times[start] < checkpoint - WINDOW_HOURS:
        start += 1
    history, times = history[start:], times[start:]
    cut = 0
    for j in range(1, len(history)):
        same = history[j]["players_tracked"] == history[0]["players_tracked"] and \
            history[j]["rates"] == history[0]["rates"]
        if not same:
            if j >= 3 and history[j]["players_tracked"] >= 1.8 * history[0]["players_tracked"]:
                cut = j - 1  # keep the head's last read as the bridge
            break
    if cut:
        history, times = history[cut:], times[cut:]
    gaps = [times[j + 1] - times[j] for j in range(len(times) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
    gaps.append(median_gap)
    weights = [max(gap, 0.01) * math.exp(-(checkpoint - t) / DECAY_HOURS)
               for gap, t in zip(gaps, times)]
    players = [float(h["players_tracked"]) for h in history]
    return times, players, [h["rates"] for h in history], weights


def forecast_players(xs, times, players, weights, x_target):
    """Players tracked at the target: damped log-log trend, capped growth."""
    last = players[-1]
    fit = wls(xs, [math.log(p) for p in players], weights) if len(xs) >= 3 else None
    if fit is None or fit[1] is None:
        target_hours = math.expm1(x_target)
        days = max(0.0, (target_hours - times[-1]) / 24.0) if times else 0.0
        return last * min(PLAYERS_GROWTH_CAP, math.exp(0.55 * math.sqrt(days)))
    ys = [math.log(p) for p in players]
    slope = fit[1] * DAMP_PLAYERS
    intercept = refit_intercept(xs, ys, weights, slope)
    grown = math.exp(intercept + slope * x_target)
    return min(last * PLAYERS_GROWTH_CAP, max(last, grown))


def cap_fit(tx, rates, tw, x_target):
    """Saturating-curve fit: rate = P / (1 + exp(-(a + b*x))).

    For each candidate cap P the transform ln(rate / (P - rate)) is linear in
    x, so a weighted line fit scores the cap; the best cap wins. Returns
    (rate_at_target, residual_sd) or None when no cap can be scored.
    """
    top = max(rates)
    if top >= 99.99:
        return None
    low = max(top * 1.03, top + 0.05)
    if low >= 100.0:
        return None
    best = None
    steps = CAP_GRID
    for k in range(steps + 1):
        cap = low * (100.0 / low) ** (k / steps)
        ys = []
        ok = True
        for rate in rates:
            r = min(max(rate, 0.005), cap * 0.999)
            ys.append(math.log(r / (cap - r)))
        fit = wls(tx, ys, tw)
        if fit is None:
            continue
        intercept, slope, resid_sd, _ = fit
        if slope is None:
            slope = 0.0
            intercept = sum(tw[j] * ys[j] for j in range(len(ys))) / sum(tw)
        sse = resid_sd
        if best is None or sse < best[0]:
            slope *= DAMP_CAP
            intercept = refit_intercept(tx, ys, tw, slope)
            y_target = intercept + slope * x_target
            rate = cap / (1.0 + math.exp(-y_target)) if y_target < 700 else cap
            best = (sse, rate, resid_sd)
    if best is None:
        return None
    return best[1], best[2]


def trophy_components(tx, rates, earners, tw, x_target, players_target,
                      group_slope_e, group_slope_l):
    """The three component forecasts for one trophy, as (rate, resid) pairs."""
    # Decomposition: extrapolate earners and divide by players at target.
    ys_e = [math.log(e + 0.5) for e in earners]
    fit_e = wls(tx, ys_e, tw)
    slope_e = shrink_slope(fit_e[1], fit_e[3], group_slope_e) * DAMP_EARNERS
    int_e = refit_intercept(tx, ys_e, tw, slope_e)
    earners_target = max(max(earners), math.exp(int_e + slope_e * x_target) - 0.5)
    earners_target = min(earners_target, players_target * 0.999)
    decomp = (100.0 * earners_target / players_target, fit_e[2])
    # Direct: the rate's logit on the same clock.
    ys_l = [logit(rate) for rate in rates]
    fit_l = wls(tx, ys_l, tw)
    slope_l = shrink_slope(fit_l[1], fit_l[3], group_slope_l) * DAMP_DIRECT
    int_l = refit_intercept(tx, ys_l, tw, slope_l)
    direct = (inv_logit(int_l + slope_l * x_target), fit_l[2])
    capped = cap_fit(tx, rates, tw, x_target)
    return capped, decomp, direct


def group_mean_slope(fits):
    """Precision-weighted mean of a group's raw slopes (0 when none exist)."""
    num = den = 0.0
    for fit in fits:
        if fit is None or fit[1] is None:
            continue
        se = fit[3] if fit[3] else 1.0
        w = 1.0 / (se * se + 0.25)
        num += w * fit[1]
        den += w
    return num / den if den > 0 else 0.0


def forecast_case(case):
    game = case.get("game") or {}
    try:
        anchor = parse_time(str(game.get("release_date")) + "T00:00:00Z")
    except (ValueError, TypeError):
        anchor = parse_time(case["history"][0]["at"])
    times, players, rates_by_read, weights = prepare_reads(case, anchor)
    target = (parse_time(case["target"]) - anchor).total_seconds() / 3600.0
    xs = [math.log1p(max(0.0, t)) for t in times]
    x_target = math.log1p(max(0.0, target))
    horizon_days = case["horizon_minutes"] / 1440.0
    players_target = forecast_players(xs, times, players, weights, x_target)
    trophies = case["trophies"]
    # Per trophy: its reads (log-hours, rates, earners, weights) and raw fits.
    per_trophy = []
    for i, trophy in enumerate(trophies):
        idx = [j for j in range(len(times)) if rates_by_read[j][i] is not None]
        tx = [xs[j] for j in idx]
        rates = [rates_by_read[j][i] for j in idx]
        earners = [rates_by_read[j][i] * players[j] / 100.0 for j in idx]
        tw = [weights[j] for j in idx]
        story = str(trophy.get("description", "")).startswith("Complete")
        fit_e = fit_l = None
        if len(idx) >= 3 and len({round(x, 6) for x in tx}) >= 2:
            fit_e = wls(tx, [math.log(e + 0.5) for e in earners], tw)
            fit_l = wls(tx, [logit(r) for r in rates], tw)
        per_trophy.append((trophy, tx, rates, earners, tw, story, fit_e, fit_l))
    mean_e = group_mean_slope([p[6] for p in per_trophy])
    mean_l = group_mean_slope([p[7] for p in per_trophy])
    story_e = [p[6] for p in per_trophy if p[5]]
    story_l = [p[7] for p in per_trophy if p[5]]
    mean_e_story = group_mean_slope(story_e) if len(story_e) >= 3 else mean_e
    mean_l_story = group_mean_slope(story_l) if len(story_l) >= 3 else mean_l
    walk = WALK_PER_ROOT_DAY * math.sqrt(horizon_days)
    zetas = [norm_ppf(tau) for tau in case["levels"]]
    rows = []
    for trophy, tx, rates, earners, tw, story, fit_e, fit_l in per_trophy:
        last_rate = rates[-1] if rates else 0.0
        if fit_e is None:
            centre, sigma = logit(last_rate), max(SIGMA_FLOOR, walk)
        else:
            g_e, g_l = (mean_e_story, mean_l_story) if story else (mean_e, mean_l)
            capped, decomp, direct = trophy_components(
                tx, rates, earners, tw, x_target, players_target, g_e, g_l)
            parts = []
            if capped is not None and W_CAP > 0:
                parts.append((W_CAP, capped))
            if W_DEC > 0:
                parts.append((W_DEC, decomp))
            if W_DIR > 0:
                parts.append((W_DIR, direct))
            if not parts:
                parts = [(1.0, decomp)]
            total_w = sum(w for w, _ in parts)
            centre = sum(w * logit(rate) for w, (rate, _) in parts) / total_w
            resid = sum(w * r for w, (_, r) in parts) / total_w
            reach = 0.4 + (x_target - tx[-1]) / 3.0
            fit_part = 0.55 * resid * min(1.5, reach)
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(walk ** 2 + fit_part ** 2)))
            if max(earners) <= 0.0:
                sigma = max(sigma, ZERO_EARNER_SIGMA)
        if last_rate >= 99.99:
            centre, sigma = logit(99.99), FULL_RATE_SIGMA
        quantiles = [round4(inv_logit(centre + z * sigma)) for z in zetas]
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
