# UFC Fight Winner Predictor

A machine-learning model that predicts the winner of a UFC fight from each
fighter's pre-fight record — striking, grappling, defense, recent form,
strength of schedule, physical attributes, and an ELO rating. Trained on
**8,600+ UFC fights (1998–2026)** and evaluated on fights it never saw during
training.

## Results

Evaluated on **all 538 UFC fights from July 2025 through July 2026** — none of
which were used in training or model selection:

| Metric | Value |
|---|---|
| **Winner accuracy** | **66.0%** |
| ROC AUC | 0.71 |
| ELO-rating baseline (pick the higher-rated fighter) | 60.0% |
| "Pick the more experienced fighter" baseline | 44.2% |

The model beats a strong ELO baseline by **+6 points** using only information
available before the fight — and it knows *when* it's likely to be right.
Accuracy climbs steadily with the model's own confidence:

| Model confidence in its pick | Share of fights | Accuracy |
|---|---|---|
| 50–65% (near coin-flips) | 40% | 57% |
| ≥65% | 60% | 72% |
| ≥70% | 35% | **77%** |
| ≥75% | 35% | **78%** |

![Accuracy by confidence](reports/accuracy_by_confidence.png)
![Calibration](reports/calibration.png)

*Left: accuracy by confidence bucket. Right: predicted probabilities line up
closely with real-world win rates (isotonic-calibrated).*

## Try it

```bash
git clone <this-repo>
cd ufc-fight-predictor
pip install -r requirements.txt

python predict.py "Islam Makhachev" "Charles Oliveira"
```

```
==========================================================
  UFC FIGHT WINNER PREDICTION
==========================================================
  Islam Makhachev            86.7%  [########################----]
  Charles Oliveira           13.3%  [####------------------------]

  Pick: Islam Makhachev  (86.7%)

  PRE-FIGHT STATS                       A           B
  UFC fights                      18          37
  Career wins                     17          25
  Win streak                      16           2
  Reach (cm)                     178         188
  Sig str/min                   2.45        3.23
  Str accuracy %                58.4        55.6
  Finish rate %                 61.1        56.8
==========================================================
```

Names are fuzzy-matched, so `python predict.py makhachev oliveira` also works.
The prediction is independent of argument order.

## How it works

- **Data** — public fighter/fight/round statistics from
  [ufcstats.com](http://ufcstats.com), via the
  [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)
  dataset. A snapshot lives in `data/`.
- **Features (69)** — every input is a *difference* between the two fighters
  ("fighter A minus fighter B"): striking output/accuracy/defense, takedowns,
  control time, finish/KO/submission rates, recent (last-5) form and its drift
  from career averages, career damage taken, strength of schedule (average
  opponent rating), physical attributes (height/reach/age/stance), and an ELO
  rating difference.
- **Strictly pre-fight** — a fighter's stats for a given bout are computed only
  from fights *before* that date, so there is no lookahead.
- **Symmetric** — each fight is learned from both fighters' perspectives, so the
  model has no "fighter A wins more often" bias.
- **Honest evaluation** — a time-based split (train `< 2024`, validate
  `2024-01 → 2025-07`, test `2025-07 →`). The model — and the ELO system's own
  update rate (`K`) — are chosen on the validation period only; the test period
  is touched exactly once, for the numbers above.
- **Calibrated** — the winning model (logistic regression) is isotonic-calibrated
  on the validation period so the probabilities mean what they say.

## Retraining / reproducing

```bash
python gen_opponent_quality.py  # regenerate strength-of-schedule table
python train.py --rebuild       # recompute features from raw data (~8 min), then train
python train.py                 # retrain from the cached feature matrix
```

`train.py` builds the feature matrix with the *same* code `predict.py` uses at
serve time, tries logistic regression / random forest / gradient boosting, picks
the best on validation AUC, calibrates it, and writes the model, metrics
(`models/metrics.json`), and the charts above. `python test_predictor.py` runs
the regression checks (no lookahead, disjoint splits, order invariance, ...).

## Notes & limitations

- MMA is high-variance — a ~66% winner-accuracy model is doing well, but any
  single fight can go either way, and the model says so with its probability.
  40% of fights are near coin-flips even to the model; most of its skill lives
  in the other 60%.
- It only knows what's in the box-score data: no injuries, camps, weight cuts,
  short-notice replacements, or stylistic intangibles.
- A handful of fighters share an identical display name in the source data
  (which doesn't disambiguate them) — for those, this tool's stats may blend
  two separate careers.
- For research and educational use. Not affiliated with or endorsed by the UFC.

## License

MIT — see [LICENSE](LICENSE). Underlying fight data belongs to ufcstats.com.
