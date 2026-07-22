"""
Regression checks for the UFC fight winner predictor. No framework — plain
asserts. Run directly:

    python test_predictor.py
"""

import json
import sys
from datetime import date

import pandas as pd

import features as F
from elo import EloRatings
from predict import predict, _load_model, _feature_row

TRAIN_CSV = "models/training_data.csv"


def test_no_lookahead():
    """A fighter's pre-fight stats as-of fight N must only see fights < N."""
    fighters, fights, rounds = F.load_data()
    fights = fights.sort_values("event_date").reset_index(drop=True)
    # Jim Miller's 4th UFC fight (vs Mac Danzig) was 2009-07-11; exactly 3 prior
    # (David Baron 2008-10-18, Matt Wiman 2008-12-10, Gray Maynard 2009-03-07).
    stats = F.compute_fighter_stats(
        "Jim Miller", fights, rounds, fighters, as_of_date=date(2009, 7, 11)
    )
    assert stats["n_ufc_fights"] == 3, f"expected 3 prior fights, got {stats['n_ufc_fights']}"
    print("PASS: no_lookahead")


def test_split_disjoint():
    """Train/val/test date windows must not overlap and must cover every row."""
    df = pd.read_csv(TRAIN_CSV, parse_dates=["event_date"])
    tr = df.event_date < "2024-01-01"
    va = (df.event_date >= "2024-01-01") & (df.event_date < "2025-07-01")
    te = df.event_date >= "2025-07-01"
    assert (tr & va).sum() == 0
    assert (va & te).sum() == 0
    assert (tr & te).sum() == 0
    assert (tr | va | te).sum() == len(df)
    print(f"PASS: split_disjoint (train={tr.sum()} val={va.sum()} test={te.sum()})")


def test_label_balance():
    """Each fight is learned from both perspectives, so focal_win should be ~50/50."""
    df = pd.read_csv(TRAIN_CSV, parse_dates=["event_date"])
    rate = df["focal_win"].mean()
    assert 0.49 <= rate <= 0.51, f"focal_win rate {rate:.4f} should be ~0.50"
    print(f"PASS: label_balance (focal_win rate = {rate:.4f})")


def test_diff_features_antisymmetric():
    """diff_* features must negate when focal/opp are swapped (A-B == -(B-A))."""
    fighters, fights, rounds = F.load_data()
    sa = F.compute_fighter_stats("Jon Jones", fights, rounds, fighters, as_of_date=date.today())
    sb = F.compute_fighter_stats("Tom Aspinall", fights, rounds, fighters, as_of_date=date.today())
    fab = F.build_features(sa, sb)
    fba = F.build_features(sb, sa)
    diff_keys = [k for k in fab if k.startswith("diff_")]
    for k in diff_keys:
        assert abs(fab[k] - (-fba[k])) < 1e-9, f"{k} not antisymmetric: {fab[k]} vs {fba[k]}"
    print(f"PASS: diff_features_antisymmetric ({len(diff_keys)} columns)")


def test_order_invariance():
    """predict.py's averaged prediction must not depend on argument order."""
    r1 = predict("Jon Jones", "Tom Aspinall")
    r2 = predict("Tom Aspinall", "Jon Jones")
    assert abs(r1["prob_a"] - r2["prob_b"]) < 1e-9
    assert abs(r1["prob_a"] + r1["prob_b"] - 1.0) < 1e-9
    print("PASS: order_invariance")


def test_feature_vector_matches_model():
    """The live feature row must match the model's expected input shape and
    produce a valid probability distribution."""
    model, feat_cols = _load_model()
    fighters, fights, rounds = F.load_data()
    elo = EloRatings(K=32).fit(fights)
    today = date.today()
    sa = F.compute_fighter_stats("Jon Jones", fights, rounds, fighters, as_of_date=today)
    sb = F.compute_fighter_stats("Tom Aspinall", fights, rounds, fighters, as_of_date=today)
    row = _feature_row("Jon Jones", "Tom Aspinall", sa, sb, elo, feat_cols, today)
    assert row.shape == (1, len(feat_cols))
    proba = model.predict_proba(row)
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-9
    print("PASS: feature_vector_matches_model")


def test_metrics_beat_baselines():
    """The committed headline numbers must actually beat the non-ML baselines."""
    m = json.loads(open("models/metrics.json").read())
    assert m["test_accuracy"] > m["baseline_elo_accuracy"]
    assert m["test_accuracy"] > m["baseline_experience_accuracy"]
    assert 0.5 < m["test_accuracy"] < 1.0
    print(f"PASS: metrics_beat_baselines (acc={m['test_accuracy']}, elo={m['baseline_elo_accuracy']})")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
