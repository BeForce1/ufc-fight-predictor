"""
UFC fight winner predictor — command-line tool.

    python predict.py "Islam Makhachev" "Charles Oliveira"
    python predict.py makhachev gaethje         # unique last names resolve too

Prints each fighter's win probability plus the pre-fight stat comparison the
model saw. Name matching is case- and accent-insensitive; an ambiguous last
name (e.g. "oliveira") lists the candidates. Stats are computed as of today
from each fighter's UFC history.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import features as F
from elo import EloRatings
from wrappers import IsotonicCalibratedModel  # noqa: F401  (needed for joblib unpickle)

MODELS_DIR = Path(__file__).resolve().parent / "models"


def _load_model():
    model = joblib.load(MODELS_DIR / "model.joblib")
    feat_cols = json.loads((MODELS_DIR / "feature_cols.json").read_text())
    elo_k = json.loads((MODELS_DIR / "elo_k.json").read_text())["K"]
    return model, feat_cols, elo_k


def _feature_row(foc_name, opp_name, foc_stats, opp_stats, elo, feat_cols, as_of):
    feats = F.build_features(foc_stats, opp_stats)
    F._populate_sos_features(feats, foc_name, opp_name, as_of=as_of)
    row = elo.add_features(pd.DataFrame([{
        "focal_fighter": foc_name, "opp_fighter": opp_name,
        "event_date": pd.Timestamp(as_of),
    }]))
    feats["elo_diff"] = float(row["elo_diff"].iloc[0])
    feats["rd_diff"]  = float(row["rd_diff"].iloc[0])
    return np.array([[float(feats.get(c, 0.0)) for c in feat_cols]])


def predict(name_a: str, name_b: str) -> dict:
    model, feat_cols, elo_k = _load_model()
    fighters, fights, rounds = F.load_data()
    elo = EloRatings(K=elo_k).fit(fights)

    row_a = F.find_fighter(name_a, fighters, fights)
    row_b = F.find_fighter(name_b, fighters, fights)
    if row_a is None:
        raise ValueError(f"Fighter not found: '{name_a}'")
    if row_b is None:
        raise ValueError(f"Fighter not found: '{name_b}'")
    a, b = row_a["fighter_name"], row_b["fighter_name"]

    today = date.today()
    sa = F.compute_fighter_stats(a, fights, rounds, fighters, as_of_date=today)
    sb = F.compute_fighter_stats(b, fights, rounds, fighters, as_of_date=today)

    # Average both perspectives so the result is independent of argument order.
    p_ab = float(model.predict_proba(_feature_row(a, b, sa, sb, elo, feat_cols, today))[0, 1])
    p_ba = float(model.predict_proba(_feature_row(b, a, sb, sa, elo, feat_cols, today))[0, 1])
    prob_a = (p_ab + (1.0 - p_ba)) / 2.0

    return {"a": a, "b": b, "prob_a": prob_a, "prob_b": 1.0 - prob_a,
            "stats_a": sa, "stats_b": sb}


def _bar(p, width=28):
    f = round(p * width)
    return "#" * f + "-" * (width - f)


def _print(r):
    a, b, pa, pb = r["a"], r["b"], r["prob_a"], r["prob_b"]
    sa, sb = r["stats_a"], r["stats_b"]
    print("\n" + "=" * 58)
    print("  UFC FIGHT WINNER PREDICTION")
    print("=" * 58)
    print(f"  {a[:24]:24s}  {pa:6.1%}  [{_bar(pa)}]")
    print(f"  {b[:24]:24s}  {pb:6.1%}  [{_bar(pb)}]")
    print(f"\n  Pick: {(a if pa >= pb else b)}  ({max(pa, pb):.1%})")
    print("\n  PRE-FIGHT STATS" + " " * 12 + f"{'A':>12}{'B':>12}")
    rows = [
        ("UFC fights",     "n_ufc_fights",   "d"),
        ("Career wins",    "career_n_wins",  ".0f"),
        ("Career losses",  "career_n_losses", ".0f"),
        ("Win streak",     "win_streak",     ".0f"),
        ("Height (cm)",    "height_cm",      ".0f"),
        ("Reach (cm)",     "reach_cm",       ".0f"),
        ("Age",            "age",            ".1f"),
        ("Sig str/min",    "sig_str_landed_pm", ".2f"),
        ("Str accuracy %", "sig_str_accuracy",  ".1f"),
        ("TD avg/15m",     "td_avg",         ".2f"),
        ("Finish rate %",  "finish_rate",    ".1f"),
    ]
    for label, key, fmt in rows:
        def _f(v):
            if v is None:
                return "N/A"
            try:
                return f"{v:{fmt}}"
            except Exception:
                return str(v)
        print(f"  {label:22s}{_f(sa.get(key)):>12}{_f(sb.get(key)):>12}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python predict.py "Fighter A" "Fighter B"')
        sys.exit(1)
    try:
        _print(predict(sys.argv[1], sys.argv[2]))
    except ValueError as e:
        print(f"\nError: {e}\nTip: try a more specific name, or last name only.")
        sys.exit(1)
