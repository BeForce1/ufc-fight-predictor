"""
Generate data/opponent_quality.csv — strength-of-schedule features.

For each (fighter, fight date), emits cumulative pre-fight descriptions of who
the fighter has faced so far, using each opponent's ELO rating *as they walked
into that bout* (causal — no future information):

    opp_elo_mean          avg rating of all prior opponents
    opp_elo_mean_wins     avg rating of opponents this fighter beat
    opp_elo_mean_losses   avg rating of opponents this fighter lost to
    wins_vs_top           wins over highly-rated opponents (>= ELO_TOP)
    losses_vs_top         losses to highly-rated opponents
    losses_vs_bottom      losses to low-rated opponents (< ELO_BOT)
    bad_loss_ratio        losses_vs_bottom / (losses + 1)

All values exclude the current fight (cumsum-shift), so a row at date D is
knowable strictly before D.

    python gen_opponent_quality.py
"""

import numpy as np
import pandas as pd

from elo import EloRatings
from features import DATA_DIR

# Roughly top/bottom quartile of active fighters, symmetric around 1500.
ELO_TOP = 1600.0
ELO_BOT = 1475.0
# K for the internal rating pass; validated against the 2024-01..2025-07
# validation period alongside the main model's K.
K = 256


def run() -> None:
    fights = pd.read_csv(DATA_DIR / "ufcstats_fights.csv", parse_dates=["event_date"])
    fights = fights.sort_values("event_date").reset_index(drop=True)
    print(f"  {len(fights):,} fights loaded; fitting ELO (K={K}) ...")
    elo = EloRatings(K=K).fit(fights)

    base = fights[["event_date", "fighter_a", "fighter_b", "outcome_a", "outcome_b"]]
    a = base.rename(columns={"fighter_a": "fighter", "fighter_b": "opp", "outcome_a": "result"})
    b = base.rename(columns={"fighter_b": "fighter", "fighter_a": "opp", "outcome_b": "result"})
    ff = pd.concat([a[["event_date", "fighter", "opp", "result"]],
                    b[["event_date", "fighter", "opp", "result"]]], ignore_index=True)

    # Opponent's rating going INTO this bout (exact-date lookup = pre-fight).
    ff["opp_elo"] = [elo.get_elo_before(str(o), d) for o, d in zip(ff.opp, ff.event_date)]
    ff["is_win"]  = (ff.result == "W").astype(int)
    ff["is_loss"] = (ff.result == "L").astype(int)
    ff["win_top"]  = ff.is_win  * (ff.opp_elo >= ELO_TOP)
    ff["loss_top"] = ff.is_loss * (ff.opp_elo >= ELO_TOP)
    ff["loss_bot"] = ff.is_loss * (ff.opp_elo <  ELO_BOT)
    ff["elo_w"] = np.where(ff.is_win  == 1, ff.opp_elo, 0.0)
    ff["elo_l"] = np.where(ff.is_loss == 1, ff.opp_elo, 0.0)

    ff = ff.sort_values(["fighter", "event_date"]).reset_index(drop=True)
    g = ff.groupby("fighter")
    def cs(col):  # cumulative sum EXCLUDING the current fight
        return g[col].transform(lambda x: x.cumsum().shift(1).fillna(0))

    n_fights = g.cumcount().astype(float)
    n_wins, n_losses = cs("is_win"), cs("is_loss")
    oq = pd.DataFrame({
        "fighter":    ff.fighter,
        "event_date": ff.event_date.dt.date,
        "opp_elo_mean":        np.where(n_fights > 0, cs("opp_elo") / n_fights.replace(0, np.nan), np.nan),
        "opp_elo_mean_wins":   np.where(n_wins   > 0, cs("elo_w") / n_wins.replace(0, np.nan),   np.nan),
        "opp_elo_mean_losses": np.where(n_losses > 0, cs("elo_l") / n_losses.replace(0, np.nan), np.nan),
        "wins_vs_top":      cs("win_top").astype(int),
        "losses_vs_top":    cs("loss_top").astype(int),
        "losses_vs_bottom": cs("loss_bot").astype(int),
        "bad_loss_ratio":   cs("loss_bot") / (n_losses + 1),
    })
    # Same-day duplicates (old one-night tournaments): keep the FIRST row —
    # the state before any of that day's fights — matching the strictly-before
    # semantics used everywhere else in the pipeline.
    oq = oq.drop_duplicates(subset=["fighter", "event_date"], keep="first")

    # Per-fighter "current" sentinel row: the full post-last-fight cumulative
    # state, dated one day after their last bout. Historical fight-date rows
    # (used to build the training matrix) are strictly pre-fight; without this
    # row, a live prediction would read the fighter's state as of *before*
    # their most recent bout. Sentinel dates can never collide with a real
    # fight date (they're strictly after each fighter's latest one).
    cur = ff.groupby("fighter").agg(
        event_date=("event_date", "max"),
        n_fights=("is_win", "size"),
        n_wins=("is_win", "sum"), n_losses=("is_loss", "sum"),
        elo_sum=("opp_elo", "sum"), elo_w=("elo_w", "sum"), elo_l=("elo_l", "sum"),
        win_top=("win_top", "sum"), loss_top=("loss_top", "sum"),
        loss_bot=("loss_bot", "sum"),
    ).reset_index()
    sentinel = pd.DataFrame({
        "fighter":    cur.fighter,
        "event_date": (pd.to_datetime(cur.event_date) + pd.Timedelta(days=1)).dt.date,
        "opp_elo_mean":        cur.elo_sum / cur.n_fights,
        "opp_elo_mean_wins":   np.where(cur.n_wins   > 0, cur.elo_w / cur.n_wins.replace(0, np.nan),   np.nan),
        "opp_elo_mean_losses": np.where(cur.n_losses > 0, cur.elo_l / cur.n_losses.replace(0, np.nan), np.nan),
        "wins_vs_top":      cur.win_top.astype(int),
        "losses_vs_top":    cur.loss_top.astype(int),
        "losses_vs_bottom": cur.loss_bot.astype(int),
        "bad_loss_ratio":   cur.loss_bot / (cur.n_losses + 1),
    })
    oq = pd.concat([oq, sentinel], ignore_index=True).sort_values(["fighter", "event_date"])

    out = DATA_DIR / "opponent_quality.csv"
    oq.to_csv(out, index=False)
    print(f"  wrote {len(oq):,} rows ({len(sentinel):,} current-state sentinels) -> {out}")


if __name__ == "__main__":
    run()
