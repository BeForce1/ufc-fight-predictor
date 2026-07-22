"""
Train the UFC fight winner predictor.
================================================================
1. Build a differential feature matrix for every UFC fight (both fighters'
   perspectives), using the *same* feature code the CLI uses at serve time.
   ELO's K is tuned against the validation period (ELO-only AUC) and the
   chosen value is saved to models/elo_k.json so predict.py fits the exact
   same ELO model at serve time.
2. Split by date (train < 2024, validation 2024-01..2025-07, test 2025-07+).
   Never touch the test set during model selection.
3. Fit LR / RandomForest / gradient-boosting, pick the best on validation AUC,
   isotonic-calibrate it, and report held-out test accuracy vs simple baselines.
4. Save the model, its feature list, metrics, and two report charts.

    python train.py            # uses cached feature matrix if present
    python train.py --rebuild  # recompute the feature matrix from raw (~8 min)
"""

import argparse
import json
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss

import features as F
from elo import EloRatings
from wrappers import IsotonicCalibratedModel

MODELS_DIR = F.DATA_DIR.parent / "models"
REPORTS_DIR = F.DATA_DIR.parent / "reports"
CACHE = MODELS_DIR / "training_data.csv"

VAL_START  = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-07-01")
SEED = 42


# ── 1. Build feature matrix (both perspectives, strictly pre-fight) ────────────

ELO_K_GRID = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]


def build_matrix() -> pd.DataFrame:
    fighters, fights, rounds = F.load_data()
    fights = fights.sort_values("event_date").reset_index(drop=True)

    rows = []
    n = len(fights)
    for i, fr in enumerate(fights.itertuples(index=False)):
        if i % 1000 == 0:
            print(f"  building features: {i:,}/{n:,}")
        oa = str(getattr(fr, "outcome_a", "")).strip().upper()
        if oa == "W":
            a_won = 1
        elif oa == "L":
            a_won = 0
        else:
            continue  # draw / no-contest: no winner to predict

        A, B = fr.fighter_a, fr.fighter_b
        D = pd.Timestamp(fr.event_date)
        d = D.date()
        sa = F.compute_fighter_stats(A, fights, rounds, fighters, as_of_date=d)
        sb = F.compute_fighter_stats(B, fights, rounds, fighters, as_of_date=d)

        for foc, opp, foc_name, opp_name, label in (
            (sa, sb, A, B, a_won),
            (sb, sa, B, A, 1 - a_won),
        ):
            feats = F.build_features(foc, opp)
            F._populate_sos_features(feats, foc_name, opp_name, as_of=d)
            feats["focal_fighter"] = foc_name
            feats["opp_fighter"]   = opp_name
            feats["event_date"]    = D
            feats["focal_win"]     = label
            feats["n_focal"]       = foc["n_ufc_fights"]
            feats["n_opp"]         = opp["n_ufc_fights"]
            rows.append(feats)

    df = pd.DataFrame(rows)

    # Tune ELO's K on the validation period only (never test) — the model
    # itself doesn't get re-trained per K here, just the ELO-only signal
    # measured against validation, which is what tune_K optimizes for.
    print("  tuning ELO K on validation period ...")
    tr_mask = df["event_date"] < VAL_START
    va_mask = (df["event_date"] >= VAL_START) & (df["event_date"] < TEST_START)
    elo = EloRatings.tune_K(fights, df[tr_mask], df[va_mask], k_values=ELO_K_GRID)
    val_probe = elo.add_features(df[va_mask])
    elo_val_auc = roc_auc_score(
        df.loc[va_mask, "focal_win"],
        elo.predict_proba_from_diff(val_probe["elo_diff"].fillna(0).values),
    )
    print(f"  selected ELO K={elo.K} (ELO-only val AUC {elo_val_auc:.4f})")
    (MODELS_DIR / "elo_k.json").write_text(json.dumps({"K": elo.K}))

    df = elo.add_features(df)   # adds elo_diff, rd_diff (as-of each fight, no lookahead)
    return df


def load_matrix(rebuild: bool) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        print(f"  using cached feature matrix {CACHE.name}")
        return pd.read_csv(CACHE, parse_dates=["event_date"])
    print("  building feature matrix from raw data (~8 min) ...")
    df = build_matrix()
    MODELS_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


# ── 2. Train / select / calibrate ─────────────────────────────────────────────

def main(rebuild: bool):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_matrix(rebuild)
    feat_cols = F.select_feature_columns(df.iloc[0].to_dict())
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)   # 0-fill matches serve
    y = df["focal_win"].astype(int)
    dt = df["event_date"]

    tr = dt < VAL_START
    va = (dt >= VAL_START) & (dt < TEST_START)
    te = dt >= TEST_START
    print(f"  split - train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")

    candidates = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=SEED),
        "HistGB": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=SEED, n_jobs=-1)
    except ImportError:
        print("  (xgboost not installed - skipping)")

    Xtr, Xva, Xte = X[tr].values, X[va].values, X[te].values
    ytr, yva = y[tr].values, y[va].values

    best_name, best_auc, best_model = None, -1.0, None
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        p_val = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, p_val)
        print(f"    {name:14s} val AUC {auc:.4f}")
        if auc > best_auc:
            best_name, best_auc, best_model = name, auc, model

    print(f"  selected {best_name} (val AUC {best_auc:.4f})")

    # Isotonic-calibrate on validation, then freeze. Isotonic (not Platt) because
    # the base scores are overconfident at the extremes — a monotonic sigmoid
    # can't correct that, a non-parametric isotonic map can.
    p_val_raw = best_model.predict_proba(Xva)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, yva)
    cal = IsotonicCalibratedModel(best_model, iso)

    # ── 3. Held-out test metrics ──────────────────────────────────────────────
    p_te = cal.predict_proba(Xte)[:, 1]
    pred = (p_te >= 0.5).astype(int)

    # Deployed semantics: each fight appears twice (both perspectives); the CLI
    # averages them into one order-invariant probability. Score that too.
    te_df = df.loc[te, ["focal_fighter", "opp_fighter", "event_date", "focal_win"]].copy()
    te_df["p"] = p_te
    te_df["pair"] = [tuple(sorted((a, b))) + (d,) for a, b, d in
                     zip(te_df.focal_fighter, te_df.opp_fighter, te_df.event_date)]
    pf_probs, pf_labels = [], []
    for _, grp in te_df.groupby("pair"):
        if len(grp) != 2:
            continue
        r0, r1 = grp.iloc[0], grp.iloc[1]
        pf_probs.append((r0.p + (1.0 - r1.p)) / 2.0)
        pf_labels.append(int(r0.focal_win))
    pf_probs, pf_labels = np.array(pf_probs), np.array(pf_labels)
    pf_acc = float(accuracy_score(pf_labels, (pf_probs >= 0.5).astype(int)))
    pf_auc = float(roc_auc_score(pf_labels, pf_probs))
    elo_k = json.loads((MODELS_DIR / "elo_k.json").read_text())["K"]
    metrics = {
        "model": best_name,
        "elo_k": elo_k,
        "trained": datetime.now().strftime("%Y-%m-%d"),
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "test_period": f"{TEST_START.date()} .. {dt.max().date()}",
        "test_accuracy": round(float(accuracy_score(y[te], pred)), 4),
        "test_auc": round(float(roc_auc_score(y[te], p_te)), 4),
        "test_perfight_accuracy": round(pf_acc, 4),
        "test_perfight_auc": round(pf_auc, 4),
        "n_test_fights": int(len(pf_labels)),
        "test_brier": round(float(brier_score_loss(y[te], p_te)), 4),
        "test_logloss": round(float(log_loss(y[te], p_te)), 4),
        "val_auc": round(float(best_auc), 4),
        "n_features": len(feat_cols),
    }

    # Simple non-ML baselines on the same test set
    elo_pred = (df.loc[te, "elo_diff"] > 0).astype(int)
    exp_pred = (df.loc[te, "n_focal"] > df.loc[te, "n_opp"]).astype(int)
    metrics["baseline_elo_accuracy"]        = round(float(accuracy_score(y[te], elo_pred)), 4)
    metrics["baseline_experience_accuracy"] = round(float(accuracy_score(y[te], exp_pred)), 4)

    print("\n  -- HELD-OUT TEST --")
    for k, v in metrics.items():
        print(f"    {k:28s} {v}")

    # ── 4. Save artifacts + charts ────────────────────────────────────────────
    joblib.dump(cal, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "feature_cols.json").write_text(json.dumps(feat_cols, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Charts use the deployed per-fight probabilities (each fight scored once,
    # both perspectives averaged) so they match the headline per-fight metrics.
    _plot_calibration(pf_labels, pf_probs, REPORTS_DIR / "calibration.png")
    _plot_accuracy_by_confidence(pf_labels, pf_probs, REPORTS_DIR / "accuracy_by_confidence.png")
    _plot_coefficients(best_model, feat_cols, REPORTS_DIR / "coefficients.png")
    print(f"\n  saved model + metrics to {MODELS_DIR}, charts to {REPORTS_DIR}")


# ── Charts ─────────────────────────────────────────────────────────────────────

COEF_LABELS = {
    "age_diff": "Age difference",
    "diff_career_n_fights": "Career fight count",
    "diff_career_total_fight_min": "Career fight-minutes",
    "diff_career_n_losses": "Career losses",
    "diff_career_abs_absorbed": "Career damage absorbed",
    "elo_diff": "ELO rating",
    "diff_career_n_wins": "Career wins",
    "diff_career_years_in_ufc": "Years in UFC",
    "diff_career_n_title_bouts": "Title bouts fought",
    "diff_career_n_title_wins": "Title bouts won",
    "diff_ufc_win_rate_last5": "Win rate, last 5",
    "diff_ufc_dec_rate": "Decision rate",
    "diff_ufc_sig_str_defense_last5": "Strike defense, last 5",
    "diff_ufc_sig_str_defense": "Strike defense, career",
    "diff_ufc_win_streak": "Current win streak",
}


def _plot_coefficients(model, feat_cols, path):
    """Top-15 standardized coefficients of the winning model. No-op if the
    selected model isn't the LR pipeline (tree models expose no linear coefs)."""
    lr = getattr(model, "named_steps", {}).get("logisticregression")
    if lr is None:
        return
    coef = lr.coef_.ravel()
    order = np.argsort(-np.abs(coef))[:15]
    names = [COEF_LABELS.get(feat_cols[i], feat_cols[i]) for i in order][::-1]
    vals  = [float(coef[i]) for i in order][::-1]
    pos, neg = "#2c3e50", "#a7b0b8"
    colors = [pos if v > 0 else neg for v in vals]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=colors, height=0.72)
    ax.axvline(0, color="#666", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=10)
    for yi, v in zip(yy, vals):
        ax.text(v + (0.006 if v > 0 else -0.006), yi, f"{v:+.3f}",
                va="center", ha="left" if v > 0 else "right", fontsize=8.5, color="#222")
    lim = max(0.42, max(abs(v) for v in vals) * 1.25)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standardized coefficient  (← favors opponent   favors fighter →)")
    ax.set_title("What the model weights most (top 15 of 69 features)")
    ax.spines[["top", "right"]].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=pos), plt.Rectangle((0, 0), 1, 1, color=neg)]
    ax.legend(h, ["raises win probability", "lowers win probability"],
              loc="upper right", fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_calibration(y_true, p, path):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 10:
            xs.append(p[m].mean())
            ys.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    ax.plot(xs, ys, "o-", color="#c0392b", label="model")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration (held-out test fights)")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_accuracy_by_confidence(y_true, p, path):
    conf = np.abs(p - 0.5) + 0.5          # confidence of the pick
    pred = (p >= 0.5).astype(int)
    correct = (pred == y_true).astype(int)
    edges = [0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["50-60%", "60-70%", "70-80%", "80%+"]
    accs, ns = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        accs.append(correct[m].mean() if m.sum() else 0.0)
        ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color="#2c3e50")
    for bar, n, a in zip(bars, ns, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.01,
                f"{a:.0%}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", color="#999")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model confidence in its pick")
    ax.set_title("Accuracy by prediction confidence (held-out test)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="recompute feature matrix from raw")
    main(ap.parse_args().rebuild)
