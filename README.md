# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hesperastryn/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hesperastryn/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI trophy on PlayStation 5: given the
house's recording of the game's Exophase trophy page up to 2026-11-21T11:30Z,
it predicts each trophy's earn rate — the percent of Exophase's tracked
players who have earned it — at 2026-11-26T00:00Z, as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by mean pinball loss on the logit
scale. `main.py` is the whole entry: Python 3.13, standard library only, no
network, deterministic, ~0.2 s a case.

## The model

Three forecasts per trophy, blended 50/30/20 on the logit scale, all fitted
by time-weighted least squares on the clock `x = ln(1 + hours since release)`
over the last 24 h of reads (a read weighs its gap to the next, decayed with
a 12 h scale; the stale pre-refresh head of the recording is dropped):

1. **A saturating curve** — `rate = P / (1 + exp(-(a + b·x)))`, cap `P`
   picked from a grid by fit quality. Earn rates bend: a chapter trophy
   piles up fast while the crowd reaches it, then flattens near the share
   who will ever get there in week one. A straight line on any scale
   overshoots that bend.
2. **A decomposition** — earners and players tracked extrapolated
   separately (log-log lines, slopes shrunk toward their group means and
   damped) and divided. This is the piece that sees dilution: a trophy
   whose earners stall while the tracked crowd keeps growing gets rarer
   even though nobody un-earns it.
3. **A direct logit line** on the same clock, shrunk and damped — the
   anchor when the other two disagree.

The spread is a random-walk term (0.35 logit units per root day) combined in
quadrature with a fit-uncertainty term from the components' residuals,
floored at 0.40 and capped at 2.00. A trophy with no earners yet gets a
spread of at least 1.10; the opening trophy every tracked player already
has is centred on 99.99 — the score's own clamp — with a 0.45 spread.

## Evidence it is built on

- The recording itself: the launch-night page sat unchanged at 638 tracked
  players (a pre-release crowd in which almost every trophy already had a
  handful of earners), then Exophase's first refresh jumped to 3,868
  players and nearly every rate fell — the dilution the decomposition
  exists to model.
- The live page at 2026-11-20T14:52Z: exactly 10,000 players tracked
  (every printed rate is an exact earner count at that N). One day in,
  Chapter 1 stood at 11.10% and the fishing-challenge trophy at 12.76%,
  while most of the 55 trophies sat at 0.01–0.03% — a steep progression
  gradient with a long, near-empty tail.
- Long-run tracker data for GTA V shows the same shape settled: the first
  story trophy at ~73% of owners, the final mission at ~17%, 100%
  completion at ~0.2% and the platinum at ~0.1%. Week one is that curve in
  fast motion, which is why the model bends instead of running straight.

## Validation

No local grade exists for the final's data, so the model was tuned on
synthetic final-shaped cases: three invented ground truths (logistic in
log-time, Weibull in time, power-law earners over saturating players), each
pinned to the two observed epochs above, read every 30 minutes with
Exophase-style staleness. Mean pinball loss over nine such cases: this
model **0.279**, the house curve baseline **0.328**, the house persistence
baseline **0.458**. On the kit's practice cases (one stale page repeated)
every fit is flat and the program degrades to persistence with an honest
spread, as it should.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

(with the competition kit's `input/` beside it; the grader runs it with
`/input` and `/output` instead). It reads every `*.in` case in name order
and writes one `<id>.out` per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9sbzt2n6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
