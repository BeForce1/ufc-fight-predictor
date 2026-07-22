"""
Feature computation for the UFC fight winner predictor.
================================================================
Given a fighter name and an as-of date, computes that fighter's pre-fight
career profile from their UFC history (strictly fights *before* the date, so
there is no lookahead). `build_features` then turns two such profiles into the
differential feature vector the model consumes ("fighter A minus fighter B").

The exact same functions are used to build the training matrix and to serve a
live prediction, so the reported accuracy is the accuracy of the CLI.
"""

import re
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from elo import DEFAULT_ELO

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Weight-class ordinal encoding (longer keys first, e.g. "light heavyweight") ─
WEIGHT_CLASS_ORDER = {
    "super heavyweight": 10,
    "light heavyweight":  8,
    "strawweight":        1,
    "flyweight":          2,
    "bantamweight":       3,
    "featherweight":      4,
    "lightweight":        5,
    "welterweight":       6,
    "middleweight":       7,
    "heavyweight":        9,
}


def _encode_weight_class(wc_raw) -> float:
    if wc_raw is None or (isinstance(wc_raw, float) and math.isnan(wc_raw)):
        return 0.0
    wc = re.sub(r"\b(ufc|title|bout|tournament|women'?s?)\b", "", str(wc_raw).lower()).strip()
    for key, val in WEIGHT_CLASS_ORDER.items():
        if key in wc:
            return float(val)
    return 0.0


PHYS_FEATS = ["height_diff", "reach_diff", "age_diff", "stance_mismatch", "weight_class_enc"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    fighters = pd.read_csv(DATA_DIR / "ufcstats_fighters.csv", parse_dates=["dob"])
    fights   = pd.read_csv(DATA_DIR / "ufcstats_fights.csv",   parse_dates=["event_date"])
    rounds   = pd.read_csv(DATA_DIR / "ufcstats_rounds.csv")
    return fighters, fights, rounds


# ── Fuzzy name matching ───────────────────────────────────────────────────────

def _norm(name: str) -> str:
    import unicodedata
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


def find_fighter(query: str, fighters_df: pd.DataFrame, fights_df: pd.DataFrame | None = None) -> dict | None:
    """
    Return best-matching fighter row as a dict, or None.
    When multiple last-name matches exist, prefer the fighter with more UFC fights.
    Raises ValueError listing candidates if ambiguous and cannot auto-resolve.
    """
    q = _norm(query)
    fighters_df = fighters_df.copy()
    fighters_df["_norm"] = fighters_df["fighter_name"].apply(_norm)

    exact = fighters_df[fighters_df["_norm"] == q]
    if len(exact) == 1:
        return exact.iloc[0].to_dict()
    if len(exact) > 1:
        return exact.sort_values("fighter_name", key=lambda s: s.str.len()).iloc[0].to_dict()

    if " " not in q:
        last_matches = fighters_df[
            fighters_df["_norm"].apply(lambda n: n.split()[-1] == q if n.split() else False)
        ]
    else:
        last_matches = fighters_df[fighters_df["_norm"].str.endswith(" " + q)]

    if len(last_matches) == 1:
        return last_matches.iloc[0].to_dict()
    if len(last_matches) > 1:
        if fights_df is not None:
            fight_counts = (
                pd.concat([fights_df["fighter_a"], fights_df["fighter_b"]])
                  .value_counts()
                  .rename("n_fights")
                  .reset_index()
                  .rename(columns={"index": "fighter_name"})
            )
            last_matches = last_matches.merge(fight_counts, on="fighter_name", how="left")
            last_matches["n_fights"] = last_matches["n_fights"].fillna(0)
            top = last_matches.sort_values("n_fights", ascending=False)
            if len(top) >= 2 and top.iloc[0]["n_fights"] > 3 * max(top.iloc[1]["n_fights"], 1):
                return top.iloc[0].to_dict()
        names = last_matches["fighter_name"].tolist()
        raise ValueError(f"Multiple fighters match '{query}': {names}\nPlease use a more specific name.")

    sub_matches = fighters_df[fighters_df["_norm"].str.contains(q, regex=False)]
    if len(sub_matches) == 1:
        return sub_matches.iloc[0].to_dict()
    if len(sub_matches) > 1:
        tokens_q = set(q.split())
        def _score(row):
            tokens_r = set(row["_norm"].split())
            return len(tokens_q & tokens_r) / max(len(tokens_q | tokens_r), 1)
        sub_matches = sub_matches.copy()
        sub_matches["_score"] = sub_matches.apply(_score, axis=1)
        best = sub_matches.sort_values("_score", ascending=False).iloc[0]
        if best["_score"] > 0.5:
            return best.to_dict()
        names = sub_matches["fighter_name"].head(5).tolist()
        raise ValueError(f"Ambiguous name '{query}'. Did you mean one of: {names}?")

    return None


# ── Fighter stat computation (strictly pre-fight) ───────────────────────────────

def compute_fighter_stats(
    fighter_name: str,
    fights_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    as_of_date: date | None = None,
) -> dict:
    """Cumulative performance stats from all UFC fights strictly before as_of_date."""
    if as_of_date is None:
        as_of_date = date.today()

    mask = (
        ((fights_df["fighter_a"] == fighter_name) | (fights_df["fighter_b"] == fighter_name)) &
        (fights_df["event_date"].dt.date < as_of_date)
    )
    fighter_fights = fights_df[mask].copy().sort_values("event_date")

    n_fights = len(fighter_fights)

    phys = fighters_df[fighters_df["fighter_name"] == fighter_name]
    if phys.empty:
        height_cm, reach_cm, stance, dob = None, None, None, None
    else:
        row = phys.iloc[0]
        height_cm = row.get("height_cm")
        reach_cm  = row.get("reach_cm")
        stance    = row.get("stance")
        dob       = row.get("dob")

    age = None
    if dob is not None and pd.notna(dob):
        try:
            dob_date = pd.Timestamp(dob).date()
            age = (as_of_date - dob_date).days / 365.25
        except Exception:
            pass

    if n_fights == 0:
        return {
            "name": fighter_name, "n_ufc_fights": 0,
            "height_cm": float(height_cm) if pd.notna(height_cm) else None,
            "reach_cm": float(reach_cm) if pd.notna(reach_cm) else None,
            "stance": stance, "age": age, "weight_class_enc": 0.0,
        }

    _fight_keys = fighter_fights[["event_name", "bout"]].drop_duplicates()
    fighter_rounds = rounds_df.merge(_fight_keys, on=["event_name", "bout"])
    fighter_rounds = fighter_rounds[fighter_rounds["fighter"] == fighter_name]

    weight_class_raw = fighter_fights["weight_class"].iloc[-1] if n_fights > 0 else None
    weight_class_enc = _encode_weight_class(weight_class_raw)

    if fighter_rounds.empty:
        sig_str_landed = sig_str_att = td_landed = td_att = sub_att = ctrl_sec = 0
        distance_landed = clinch_landed = ground_landed = 0
        head_landed = body_landed = leg_landed = 0
        against_sig_str_landed = against_sig_str_att = 0
        against_td_landed = against_td_att = 0
        total_time_sec = 0
    else:
        sig_str_landed  = fighter_rounds["sig_str_landed"].sum()
        sig_str_att     = fighter_rounds["sig_str_att"].sum()
        td_landed       = fighter_rounds["td_landed"].sum()
        td_att          = fighter_rounds["td_att"].sum()
        sub_att         = fighter_rounds["sub_att"].sum()
        ctrl_sec        = fighter_rounds["ctrl_sec"].sum()
        distance_landed = fighter_rounds["distance_landed"].sum() if "distance_landed" in fighter_rounds.columns else 0
        clinch_landed   = fighter_rounds["clinch_landed"].sum()   if "clinch_landed"   in fighter_rounds.columns else 0
        ground_landed   = fighter_rounds["ground_landed"].sum()   if "ground_landed"   in fighter_rounds.columns else 0
        head_landed     = fighter_rounds["head_landed"].sum()     if "head_landed"     in fighter_rounds.columns else 0
        body_landed     = fighter_rounds["body_landed"].sum()     if "body_landed"     in fighter_rounds.columns else 0
        leg_landed      = fighter_rounds["leg_landed"].sum()      if "leg_landed"      in fighter_rounds.columns else 0

        bout_keys = fighter_rounds[["event_name", "bout"]].drop_duplicates()
        opp_rounds = rounds_df.merge(bout_keys, on=["event_name", "bout"])
        opp_rounds = opp_rounds[opp_rounds["fighter"] != fighter_name]
        against_sig_str_landed = opp_rounds["sig_str_landed"].sum() if not opp_rounds.empty else 0
        against_sig_str_att    = opp_rounds["sig_str_att"].sum()    if not opp_rounds.empty else 0
        against_td_landed      = opp_rounds["td_landed"].sum()      if not opp_rounds.empty else 0
        against_td_att         = opp_rounds["td_att"].sum()         if not opp_rounds.empty else 0

        total_time_sec = 0
        for _, frow in fighter_fights.iterrows():
            fr = int(frow.get("finish_round", 3) or 3)
            ft_str = str(frow.get("finish_time", "5:00") or "5:00")
            try:
                mins, secs = ft_str.strip().split(":")
                last_round_sec = int(mins) * 60 + int(secs)
            except Exception:
                last_round_sec = 300
            total_time_sec += (fr - 1) * 300 + last_round_sec

    total_time_min = total_time_sec / 60.0
    total_time_15  = total_time_sec / 900.0

    def _safe(num, den):
        return float(num / den) if den > 0 else None

    sig_str_landed_pm = _safe(sig_str_landed, total_time_min)
    sig_str_accuracy  = _safe(sig_str_landed * 100.0, sig_str_att)
    td_avg            = _safe(td_landed, total_time_15)
    td_accuracy       = _safe(td_landed * 100.0, td_att)
    sub_avg_stat      = _safe(sub_att, total_time_15)
    ctrl_time_avg     = _safe(ctrl_sec, total_time_15)

    sig_str_absorbed_pm = _safe(against_sig_str_landed, total_time_min)
    sig_str_defense     = (1.0 - against_sig_str_landed / against_sig_str_att) if against_sig_str_att > 0 else None
    td_defense          = (1.0 - against_td_landed / against_td_att) if against_td_att > 0 else None

    distance_rate = _safe(distance_landed * 100.0, sig_str_att)
    clinch_rate   = _safe(clinch_landed   * 100.0, sig_str_att)
    ground_rate   = _safe(ground_landed   * 100.0, sig_str_att)
    head_rate     = _safe(head_landed     * 100.0, sig_str_att)
    body_rate     = _safe(body_landed     * 100.0, sig_str_att)
    leg_rate      = _safe(leg_landed      * 100.0, sig_str_att)

    wins = losses = finishes = kos = subs = decisions = got_finished = win_streak = 0
    peak_streak = 0
    title_bouts = title_wins = 0
    last_result = None
    for _, frow in fighter_fights.iterrows():
        is_a = frow["fighter_a"] == fighter_name
        result = frow["outcome_a"] if is_a else frow["outcome_b"]
        method = str(frow.get("method", "")).upper()
        is_ko_like = ("KO" in method or "TKO" in method)
        is_sub_like = ("SUB" in method)
        is_dec = ("DECISION" in method or "DEC." in method)
        bout_label = str(frow.get("weight_class", "") or "")
        is_title = "title" in bout_label.lower()
        if is_title:
            title_bouts += 1
        if result == "W":
            wins += 1
            win_streak = win_streak + 1 if last_result == "W" else 1
            peak_streak = max(peak_streak, win_streak)
            if is_title:
                title_wins += 1
            if is_ko_like:
                kos += 1; finishes += 1
            elif is_sub_like:
                subs += 1; finishes += 1
            if is_dec:
                decisions += 1
            last_result = "W"
        elif result == "L":
            losses += 1
            win_streak = 0
            if is_ko_like or is_sub_like:
                got_finished += 1
            if is_dec:
                decisions += 1
            last_result = "L"

    total_fights_f = float(n_fights)
    finish_rate       = finishes     / total_fights_f * 100.0 if total_fights_f > 0 else None
    ko_rate           = kos          / total_fights_f * 100.0 if total_fights_f > 0 else None
    sub_rate          = subs         / total_fights_f * 100.0 if total_fights_f > 0 else None
    dec_rate          = decisions    / total_fights_f * 100.0 if total_fights_f > 0 else None
    got_finished_rate = got_finished / total_fights_f * 100.0 if total_fights_f > 0 else None

    last_fight_date = fighter_fights["event_date"].max().date() if n_fights > 0 else None
    layoff_days = float((as_of_date - last_fight_date).days) if last_fight_date else 0.0
    first_fight_date = fighter_fights["event_date"].min().date() if n_fights > 0 else None
    years_in_ufc = ((as_of_date - first_fight_date).days / 365.25) if first_fight_date else 0.0

    # ── Last-5 rolling-window form ────────────────────────────────────────────
    last5_fights = fighter_fights.tail(5)
    n_last5 = len(last5_fights)
    if n_last5 > 0 and not fighter_rounds.empty:
        last5_keys = last5_fights[["event_name", "bout"]].drop_duplicates()
        l5_rounds = rounds_df.merge(last5_keys, on=["event_name", "bout"])
        l5_self = l5_rounds[l5_rounds["fighter"] == fighter_name]
        l5_opp  = l5_rounds[l5_rounds["fighter"] != fighter_name]

        l5_sig_str_landed  = l5_self["sig_str_landed"].sum()
        l5_sig_str_att     = l5_self["sig_str_att"].sum()
        l5_td_landed       = l5_self["td_landed"].sum()
        l5_td_att          = l5_self["td_att"].sum()
        l5_ctrl_sec        = l5_self["ctrl_sec"].sum()
        l5_against_sig_str_landed = l5_opp["sig_str_landed"].sum() if not l5_opp.empty else 0
        l5_against_sig_str_att    = l5_opp["sig_str_att"].sum()    if not l5_opp.empty else 0
        l5_against_td_landed      = l5_opp["td_landed"].sum()      if not l5_opp.empty else 0
        l5_against_td_att         = l5_opp["td_att"].sum()         if not l5_opp.empty else 0

        l5_total_time_sec = 0
        for _, frow in last5_fights.iterrows():
            fr = int(frow.get("finish_round", 3) or 3)
            ft_str = str(frow.get("finish_time", "5:00") or "5:00")
            try:
                mins, secs = ft_str.strip().split(":")
                last_round_sec = int(mins) * 60 + int(secs)
            except Exception:
                last_round_sec = 300
            l5_total_time_sec += (fr - 1) * 300 + last_round_sec

        l5_total_time_min = l5_total_time_sec / 60.0
        l5_total_time_15  = l5_total_time_sec / 900.0

        l5_wins = l5_kos = l5_subs = l5_got_finished = 0
        for _, frow in last5_fights.iterrows():
            is_a = frow["fighter_a"] == fighter_name
            result = frow["outcome_a"] if is_a else frow["outcome_b"]
            method = str(frow.get("method", "")).upper()
            is_ko_like = "KO" in method or "TKO" in method
            is_sub_like = "SUB" in method
            if result == "W":
                l5_wins += 1
                if is_ko_like:
                    l5_kos += 1
                elif is_sub_like:
                    l5_subs += 1
            elif result == "L" and (is_ko_like or is_sub_like):
                l5_got_finished += 1

        last5_stats = {
            "sig_str_landed_pm_last5":   _safe(l5_sig_str_landed,        l5_total_time_min),
            "sig_str_accuracy_last5":    _safe(l5_sig_str_landed * 100., l5_sig_str_att),
            "td_avg_last5":              _safe(l5_td_landed,             l5_total_time_15),
            "td_accuracy_last5":         _safe(l5_td_landed * 100.,      l5_td_att),
            "ctrl_time_avg_last5":       _safe(l5_ctrl_sec,              l5_total_time_15),
            "win_rate_last5":            (l5_wins / n_last5 * 100.0),
            "ko_rate_last5":             (l5_kos / n_last5 * 100.0),
            "sub_rate_last5":            (l5_subs / n_last5 * 100.0),
            "got_finished_rate_last5":   (l5_got_finished / n_last5 * 100.0),
            "sig_str_absorbed_pm_last5": _safe(l5_against_sig_str_landed, l5_total_time_min),
            "sig_str_defense_last5":     (1.0 - l5_against_sig_str_landed / l5_against_sig_str_att) if l5_against_sig_str_att > 0 else None,
            "td_defense_last5":          (1.0 - l5_against_td_landed / l5_against_td_att) if l5_against_td_att > 0 else None,
        }
    else:
        last5_stats = {k: None for k in (
            "sig_str_landed_pm_last5", "sig_str_accuracy_last5", "td_avg_last5",
            "td_accuracy_last5", "ctrl_time_avg_last5", "win_rate_last5",
            "ko_rate_last5", "sub_rate_last5", "got_finished_rate_last5",
            "sig_str_absorbed_pm_last5", "sig_str_defense_last5", "td_defense_last5",
        )}

    return {
        "name":            fighter_name,
        "n_ufc_fights":    n_fights,
        "height_cm":       float(height_cm) if pd.notna(height_cm) else None,
        "reach_cm":        float(reach_cm)  if pd.notna(reach_cm)  else None,
        "stance":          stance,
        "age":             age,
        "weight_class_enc": weight_class_enc,
        "sig_str_landed_pm": sig_str_landed_pm,
        "sig_str_accuracy":  sig_str_accuracy,
        "td_avg":           td_avg,
        "td_accuracy":      td_accuracy,
        "sub_avg":          sub_avg_stat,
        "ctrl_time_avg":    ctrl_time_avg,
        "win_streak":       float(win_streak),
        "finish_rate":      finish_rate,
        "ko_rate":          ko_rate,
        "sub_rate":         sub_rate,
        "distance_rate":    distance_rate,
        "clinch_rate":      clinch_rate,
        "ground_rate":      ground_rate,
        "head_rate":        head_rate,
        "body_rate":        body_rate,
        "leg_rate":         leg_rate,
        "dec_rate":              dec_rate,
        "sig_str_absorbed_pm":   sig_str_absorbed_pm,
        "sig_str_defense":       sig_str_defense,
        "td_defense":            td_defense,
        "got_finished_rate":     got_finished_rate,
        "layoff_days":           layoff_days,
        "last_fight_date":  last_fight_date,
        **last5_stats,
        "career_n_fights":         float(n_fights),
        "career_n_wins":           float(wins),
        "career_n_losses":         float(losses),
        "career_n_finishes":       float(finishes),
        "career_total_fight_min":  float(total_time_min),
        "career_peak_win_streak":  float(peak_streak),
        "career_n_title_bouts":    float(title_bouts),
        "career_n_title_wins":     float(title_wins),
        "career_years_in_ufc":     float(years_in_ufc),
    }


# ── Feature vector construction ───────────────────────────────────────────────

def _stance_encode(stance: str | None) -> int:
    if stance is None:
        return 0
    return {"orthodox": 1, "southpaw": 2, "switch": 3, "open stance": 4}.get(
        str(stance).lower().strip(), 0
    )


_OQ_CACHE: dict = {}

# ELO-mean SOS columns are centered on DEFAULT_ELO (~1500), not 0 — a fighter
# with no qualifying prior opponents has no signal, not a literal zero. Filling
# with 0.0 would manufacture a ~1500-point spurious diff against any opponent
# who does have a real value. Count/ratio columns (wins_vs_top etc.) are
# correctly 0 when there's no history, so only these get the neutral fill.
_SOS_NEUTRAL_FILL = {
    "opp_elo_mean": DEFAULT_ELO,
    "opp_elo_mean_wins": DEFAULT_ELO,
    "opp_elo_mean_losses": DEFAULT_ELO,
}

def _populate_sos_features(feats: dict, name_a: str, name_b: str, as_of=None) -> None:
    """Add strength-of-schedule (opponent-quality) columns from opponent_quality.csv."""
    if "oq" not in _OQ_CACHE:
        oq_path = DATA_DIR / "opponent_quality.csv"
        _OQ_CACHE["oq"] = pd.read_csv(oq_path, parse_dates=["event_date"]) if oq_path.exists() else None
    oq = _OQ_CACHE["oq"]
    if oq is None:
        return

    sos_cols = [c for c in oq.columns if c not in ("fighter", "event_date")]

    def _latest(name):
        sub = oq[oq.fighter == name]
        if as_of is not None:
            sub = sub[sub.event_date <= pd.Timestamp(as_of)]
        if sub.empty:
            return {c: _SOS_NEUTRAL_FILL.get(c, 0.0) for c in sos_cols}
        row = sub.sort_values("event_date").iloc[-1]
        return {c: (float(row[c]) if pd.notna(row[c]) else _SOS_NEUTRAL_FILL.get(c, 0.0)) for c in sos_cols}

    a_vals, b_vals = _latest(name_a), _latest(name_b)
    for c in sos_cols:
        feats[f"focal_{c}"] = a_vals[c]
        feats[f"opp_{c}"]   = b_vals[c]
        feats[f"diff_{c}"]  = a_vals[c] - b_vals[c]


def build_features(focal: dict, opp: dict) -> dict:
    """Flat differential feature dict from two fighter stat dicts (focal perspective)."""
    def _v(d, key):
        val = d.get(key)
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else 0.0

    def _raw(d, key):
        """None (not 0.0) when missing — 0 is a real value for some fields."""
        val = d.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return float(val)

    def _phys_diff(key):
        """height/reach/age: missing on either side means no comparison is
        possible (0 diff), not a phantom comparison against a 0cm/0-year-old
        fighter."""
        fv, ov = _raw(focal, key), _raw(opp, key)
        return (fv - ov) if (fv is not None and ov is not None) else 0.0

    focal_stance_enc = _stance_encode(focal.get("stance"))
    opp_stance_enc   = _stance_encode(opp.get("stance"))
    stance_mismatch  = 1 if (focal_stance_enc > 0 and opp_stance_enc > 0 and focal_stance_enc != opp_stance_enc) else 0

    feats = {
        "height_diff":      _phys_diff("height_cm"),
        "reach_diff":       _phys_diff("reach_cm"),
        "age_diff":         _phys_diff("age"),
        "stance_mismatch":  stance_mismatch,
        "weight_class_enc": _v(focal, "weight_class_enc"),
    }

    rate_map = {
        "ufc_sig_str_landed_pm": "sig_str_landed_pm",
        "ufc_sig_str_accuracy":  "sig_str_accuracy",
        "ufc_td_avg":            "td_avg",
        "ufc_td_accuracy":       "td_accuracy",
        "ufc_sub_avg":           "sub_avg",
        "ufc_ctrl_time_avg":     "ctrl_time_avg",
        "ufc_win_streak":        "win_streak",
        "ufc_finish_rate":       "finish_rate",
        "ufc_ko_rate":           "ko_rate",
        "ufc_sub_rate":          "sub_rate",
        "ufc_distance_rate":     "distance_rate",
        "ufc_clinch_rate":       "clinch_rate",
        "ufc_ground_rate":       "ground_rate",
        "ufc_head_rate":         "head_rate",
        "ufc_body_rate":         "body_rate",
        "ufc_leg_rate":          "leg_rate",
        "ufc_dec_rate":              "dec_rate",
        "ufc_sig_str_absorbed_pm":   "sig_str_absorbed_pm",
        "ufc_sig_str_defense":       "sig_str_defense",
        "ufc_td_defense":            "td_defense",
        "ufc_got_finished_rate":     "got_finished_rate",
        "ufc_layoff_days":           "layoff_days",
        "ufc_sig_str_landed_pm_last5":   "sig_str_landed_pm_last5",
        "ufc_sig_str_accuracy_last5":    "sig_str_accuracy_last5",
        "ufc_td_avg_last5":              "td_avg_last5",
        "ufc_td_accuracy_last5":         "td_accuracy_last5",
        "ufc_ctrl_time_avg_last5":       "ctrl_time_avg_last5",
        "ufc_win_rate_last5":            "win_rate_last5",
        "ufc_ko_rate_last5":             "ko_rate_last5",
        "ufc_sub_rate_last5":            "sub_rate_last5",
        "ufc_got_finished_rate_last5":   "got_finished_rate_last5",
        "ufc_sig_str_absorbed_pm_last5": "sig_str_absorbed_pm_last5",
        "ufc_sig_str_defense_last5":     "sig_str_defense_last5",
        "ufc_td_defense_last5":          "td_defense_last5",
        "career_n_fights":         "career_n_fights",
        "career_n_wins":           "career_n_wins",
        "career_n_losses":         "career_n_losses",
        "career_n_finishes":       "career_n_finishes",
        "career_total_fight_min":  "career_total_fight_min",
        "career_peak_win_streak":  "career_peak_win_streak",
        "career_n_title_bouts":    "career_n_title_bouts",
        "career_n_title_wins":     "career_n_title_wins",
        "career_years_in_ufc":     "career_years_in_ufc",
    }

    for feat_col, stat_key in rate_map.items():
        f_val = _v(focal, stat_key)
        o_val = _v(opp,   stat_key)
        feats[f"focal_{feat_col}"] = f_val
        feats[f"opp_{feat_col}"]   = o_val
        feats[f"diff_{feat_col}"]  = f_val - o_val

    # Drift features (last5 - cumulative), decorrelating the redundant pair.
    DRIFT_PAIRS = [
        "ufc_sig_str_landed_pm", "ufc_sig_str_accuracy",
        "ufc_td_avg", "ufc_td_accuracy", "ufc_ctrl_time_avg",
        "ufc_ko_rate", "ufc_sub_rate", "ufc_got_finished_rate",
        "ufc_sig_str_absorbed_pm", "ufc_sig_str_defense", "ufc_td_defense",
    ]
    for base in DRIFT_PAIRS:
        f_drift = feats.get(f"focal_{base}_last5", 0.0) - feats.get(f"focal_{base}", 0.0)
        o_drift = feats.get(f"opp_{base}_last5",   0.0) - feats.get(f"opp_{base}",   0.0)
        feats[f"focal_{base}_drift"] = f_drift
        feats[f"opp_{base}_drift"]   = o_drift
        feats[f"diff_{base}_drift"]  = f_drift - o_drift

    return feats


def select_feature_columns(feats: dict) -> list[str]:
    """Deterministic column order: physical + every differential feature + ELO.

    Differential-only (each column negates when the two fighters are swapped),
    so the model learns an antisymmetric decision function.
    """
    diff_cols = sorted(c for c in feats if c.startswith("diff_"))
    return PHYS_FEATS + diff_cols + ["elo_diff", "rd_diff"]
