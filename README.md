# UFC Fight Winner Predictor

A machine-learning model that predicts the winner of a UFC fight from each
fighter's pre-fight record — striking, grappling, defense, recent form,
strength of schedule, physical attributes, and an ELO rating. Trained on
**8,600+ UFC fights (1998–2026)** and evaluated on fights it never saw during
training.

## Results

Evaluated on **every UFC fight from July 2025 onward** (1,076 fighter-perspective
predictions across ~540 bouts) — none of which were used in training or model
selection:

| Metric | Value |
|---|---|
| **Winner accuracy** | **63.8%** |
| ROC AUC | 0.663 |
| Brier score | 0.236 |
| ELO-rating baseline (pick the higher-rated fighter) | 55.2% |
| "Pick the more experienced fighter" baseline | 44.2% |

The model beats a strong ELO baseline by **+8.5 points** using only information
available before the fight. Accuracy climbs with the model's own confidence, and
its probabilities are well-calibrated:

![Accuracy by confidence](reports/accuracy_by_confidence.png)
![Calibration](reports/calibration.png)

*Left: when the model is 70–80% sure of its pick, it's right ~70% of the time.
Right: predicted probabilities line up closely with real-world win rates.*

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
  Islam Makhachev            83.3%  [#######################-----]
  Charles Oliveira           16.7%  [#####-----------------------]

  Pick: Islam Makhachev  (83.3%)

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
- **Features (68)** — every input is a *difference* between the two fighters
  ("fighter A minus fighter B"): striking output/accuracy/defense, takedowns,
  control time, finish/KO/submission rates, recent (last-5) form and its drift
  from career averages, strength of schedule (average opponent rating), physical
  edges (height/reach/age/stance), and an ELO rating difference.
- **Strictly pre-fight** — a fighter's stats for a given bout are computed only
  from fights *before* that date, so there is no lookahead.
- **Symmetric** — each fight is learned from both fighters' perspectives, so the
  model has no "fighter A wins more often" bias.
- **Honest evaluation** — a time-based split (train `< 2024`, validate
  `2024-01 → 2025-07`, test `2025-07 →`). The model is chosen on the validation
  period only; the test period is touched exactly once, for the numbers above.
- **Calibrated** — the winning model (logistic regression) is isotonic-calibrated
  on the validation period so the probabilities mean what they say.

## Retraining / reproducing

```bash
python train.py --rebuild     # recompute features from raw data (~8 min), then train
python train.py               # retrain from the cached feature matrix
```

`train.py` builds the feature matrix with the *same* code `predict.py` uses at
serve time, tries logistic regression / random forest / gradient boosting, picks
the best on validation AUC, calibrates it, and writes the model, metrics
(`models/metrics.json`), and the charts above.

## Notes & limitations

- MMA is high-variance — a ~64% winner-accuracy model is doing well, but any
  single fight can go either way, and the model says so with its probability.
- It only knows what's in the box-score data: no injuries, camps, weight cuts,
  short-notice replacements, or stylistic intangibles.
- For research and educational use. Not affiliated with or endorsed by the UFC.

## License

MIT — see [LICENSE](LICENSE). Underlying fight data belongs to ufcstats.com.
