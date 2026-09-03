"""
Side-quest analyses: things people believe about fights, tested.
================================================================
Two questions, answered from the same public UFCStats snapshot the model
trains on (`python myths.py`, ~10s, writes charts to reports/):

1. Do "cosmic" factors matter? Zodiac signs, moon phase, Mercury retrograde,
   Friday the 13th, scored against who wins and how fights end, next to a
   pre-registered noise yardstick: 200 columns of pure random noise scored
   the same way. A real signal must beat what noise achieves at the same n.

2. Do fighters gas? Per-round output (significant-strike attempts per minute)
   split by winner/loser and decision/finish, with early-ended rounds
   normalised by their true duration from `finish_time`.

Everything here is descriptive research on public box-score data; none of it
feeds the model.
"""

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
RANDOM_SEED = 42

# ── cosmic primitives ─────────────────────────────────────────────────────────

NEW_MOON_EPOCH = pd.Timestamp("2000-01-06 18:14")   # first new moon after J2000
SYNODIC = 29.530588853


def moon_age(dates):
    """Days since new moon, 0..29.53. Mean synodic month, within ~0.6d of the
    true ephemeris, plenty for a yes/no astrology question."""
    d = (pd.to_datetime(dates) - NEW_MOON_EPOCH).dt.total_seconds() / 86400.0
    return np.mod(d, SYNODIC)


def moon_illum(dates):
    return (1 - np.cos(2 * np.pi * moon_age(dates) / SYNODIC)) / 2


def mercury_retrograde(dates):
    """1 when Mercury's geocentric ecliptic longitude decreases day-over-day.
    Coplanar circular orbits, so windows land within ~2 days of the real
    ephemeris. Good enough for astrology."""
    def lon(d):
        le = np.radians(100.46435 + 0.98560028 * d)
        lm = np.radians(252.25084 + 4.09233445 * d)
        x = 0.387098 * np.cos(lm) - np.cos(le)
        y = 0.387098 * np.sin(lm) - np.sin(le)
        return np.degrees(np.arctan2(y, x))
    d = (pd.to_datetime(dates) - pd.Timestamp("2000-01-01 12:00")).dt.total_seconds() / 86400.0
    step = (lon(d + 1) - lon(d) + 180) % 360 - 180
    return (step < 0).astype(int)


_ZODIAC_CUTS = [(1, 20, "aquarius"), (2, 19, "pisces"), (3, 21, "aries"),
                (4, 20, "taurus"), (5, 21, "gemini"), (6, 21, "cancer"),
                (7, 23, "leo"), (8, 23, "virgo"), (9, 23, "libra"),
                (10, 23, "scorpio"), (11, 22, "sagittarius"), (12, 22, "capricorn")]
SIGNS = ["aries", "taurus", "gemini", "cancer", "leo", "virgo",
         "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
ELEMENT = {s: e for e, ss in {
    "fire":  ["aries", "leo", "sagittarius"],
    "earth": ["taurus", "virgo", "capricorn"],
    "air":   ["gemini", "libra", "aquarius"],
    "water": ["cancer", "scorpio", "pisces"]}.items() for s in ss}


def zodiac(ts):
    sign = "capricorn"
    for cm, cd, s in _ZODIAC_CUTS:
        if (ts.month, ts.day) >= (cm, cd):
            sign = s
    return sign


# ── scoring against a noise yardstick ─────────────────────────────────────────

def score(x, y):
    """n / AUC / Pearson r / p for one feature against one target."""
    m = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x[m], float), np.asarray(y[m], float)
    if len(x) < 30 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    r, p = stats.pearsonr(x, y)
    binary = set(np.unique(y)) <= {0.0, 1.0}
    return dict(n=len(x), auc=roc_auc_score(y, x) if binary else np.nan, r=r, p=p)


def null_band(y, n_draws=200):
    """What pure noise achieves against this target at this n. Effect is
    |AUC-0.5| for a binary target, |r| for a continuous one. Returns the 95th
    percentile and the worst of all 200 draws (the family-wise yardstick: with
    many features tested, a real signal should beat the *best* noise column)."""
    rng = np.random.default_rng(RANDOM_SEED)
    yv = np.asarray(y.dropna(), float)
    binary = set(np.unique(yv)) <= {0.0, 1.0}
    eff = []
    for _ in range(n_draws):
        z = rng.standard_normal(len(yv))
        eff.append(abs(roc_auc_score(yv, z) - 0.5) if binary else
                   abs(stats.pearsonr(z, yv)[0]))
    return float(np.percentile(eff, 95)), float(max(eff))


def effect(s):
    return abs(s["auc"] - 0.5) if not np.isnan(s["auc"]) else abs(s["r"])


# ── data assembly ─────────────────────────────────────────────────────────────

def load():
    f = pd.read_csv(DATA_DIR / "ufcstats_fights.csv", parse_dates=["event_date"])
    fighters = pd.read_csv(DATA_DIR / "ufcstats_fighters.csv", parse_dates=["dob"])
    dob = fighters.groupby("fighter_name")["dob"].first()

    mmss = f["finish_time"].astype(str).str.extract(r"^(\d+):(\d+)$").astype(float)
    fin_secs = mmss[0] * 60 + mmss[1]
    std = f["time_format"].astype(str).str.contains(r"\(5-5", regex=True, na=False)
    f["fight_secs"] = np.where(std, (f["finish_round"] - 1) * 300 + fin_secs, np.nan)

    # Deterministic focal-fighter orientation. UFCStats lists the winner first,
    # so taking fighter_a as-is would leak the result into every feature.
    flip = f["fight_url"].map(
        lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0)
    f["focal"] = np.where(flip, f["fighter_a"], f["fighter_b"])
    f["opp"] = np.where(flip, f["fighter_b"], f["fighter_a"])
    f["focal_win"] = np.where(
        (f["outcome_a"] == "W") ^ ~flip, 1.0, 0.0)
    f.loc[~f["outcome_a"].isin(["W", "L"]), "focal_win"] = np.nan  # draws / NC

    for side in ("focal", "opp"):
        b = f[side].map(dob)
        f[f"{side}_age"] = (f["event_date"] - b).dt.days / 365.25
        f[f"{side}_sign"] = b.map(lambda t: zodiac(t) if pd.notna(t) else np.nan)
    return f


def cosmic_tables(f):
    d = f["event_date"]
    # Winner side: differentials, focal minus opponent.
    sign_idx = {s: i for i, s in enumerate(SIGNS)}
    W = pd.DataFrame({
        "d_age_years": f["focal_age"] - f["opp_age"],
        "d_zodiac_sign_idx": f["focal_sign"].map(sign_idx) - f["opp_sign"].map(sign_idx),
        "d_name_length": f["focal"].str.len() - f["opp"].str.len(),
    })
    # Fight character: symmetric per-event features.
    G = pd.DataFrame({
        "moon_illumination": moon_illum(d),
        "moon_age_days": moon_age(d),
        "is_full_moon_pm1.5d": (np.abs(moon_age(d) - SYNODIC / 2) <= 1.5).astype(int),
        "is_new_moon_pm1.5d": (np.minimum(moon_age(d), SYNODIC - moon_age(d)) <= 1.5).astype(int),
        "mercury_retrograde": mercury_retrograde(d),
        "friday_the_13th": ((d.dt.day == 13) & (d.dt.dayofweek == 4)).astype(int),
        "same_zodiac_sign": (f["focal_sign"] == f["opp_sign"]).astype(float)
                            .where(f["focal_sign"].notna() & f["opp_sign"].notna()),
        "same_zodiac_element": (f["focal_sign"].map(ELEMENT) == f["opp_sign"].map(ELEMENT))
                               .astype(float)
                               .where(f["focal_sign"].notna() & f["opp_sign"].notna()),
    })
    method = f["method"].fillna("").str.lower()
    T = pd.DataFrame({
        "is_finish": (~method.str.contains("decision")).astype(float)
                     .where(f["outcome_a"].isin(["W", "L"])),
        "is_ko": method.str.contains("ko/tko").astype(float),
        "is_submission": method.str.contains("submission").astype(float),
        "fight_secs": f["fight_secs"],
    })

    rows = []
    bands = {"focal_win": null_band(f["focal_win"])}
    for c in W.columns:
        s = score(W[c], f["focal_win"])
        rows.append(dict(feature=c, target="focal_win", **s, effect=effect(s)))
    for t in T.columns:
        bands[t] = null_band(T[t])
        for c in G.columns:
            s = score(G[c], T[t])
            if s is not None:
                rows.append(dict(feature=c, target=t, **s, effect=effect(s)))
    out = pd.DataFrame(rows)
    out["noise_p95"] = out["target"].map(lambda t: bands[t][0])
    out["noise_max200"] = out["target"].map(lambda t: bands[t][1])
    out["verdict"] = np.where(out["effect"] > out["noise_max200"], "SIGNAL",
                     np.where(out["effect"] > out["noise_p95"], "inside family band", "noise"))
    return out.sort_values("effect", ascending=False).reset_index(drop=True), bands


def age_controlled(f):
    """Anything that clears the noise gate gets one more test: is it just the
    age effect wearing a costume? Residualize on age difference and re-score."""
    sign_idx = {s: i for i, s in enumerate(SIGNS)}
    d_age = f["focal_age"] - f["opp_age"]
    y = f["focal_win"]
    rows = []
    for name, x in [
            ("d_name_length", f["focal"].str.len() - f["opp"].str.len()),
            ("d_zodiac_sign_idx",
             f["focal_sign"].map(sign_idx) - f["opp_sign"].map(sign_idx))]:
        m = x.notna() & d_age.notna() & y.notna()
        resid = x[m] - np.polyval(np.polyfit(d_age[m], x[m], 1), d_age[m])
        raw, ctl = score(x[m], y[m]), score(resid, y[m])
        rows.append(dict(feature=name, raw_auc=raw["auc"], raw_r=raw["r"],
                         age_controlled_auc=ctl["auc"], age_controlled_r=ctl["r"],
                         age_controlled_p=ctl["p"]))
    return pd.DataFrame(rows)


# ── cardio: does anyone actually gas? ─────────────────────────────────────────

def cardio_rounds(f):
    """Round-level output per minute. A fight that ended early has a partial
    final round, and raw per-round volume there looks exactly like fading,
    so the final round is normalised by its real duration from `finish_time`."""
    r = pd.read_csv(DATA_DIR / "ufcstats_rounds.csv")
    cols = ["event_name", "bout", "event_date", "method", "finish_round",
            "finish_time", "time_format"]
    r = r.merge(f[cols], on=["event_name", "bout"], how="left")
    r = r.dropna(subset=["round", "sig_str_att", "event_date", "finish_round"])
    # Standard 5-minute-round formats only; the 1998-2000 10- and 12-minute
    # rounds would break the "every earlier round is 300s" assumption.
    r = r[r["time_format"].astype(str).str.contains(r"\(5-5", regex=True, na=False)]

    mmss = r["finish_time"].astype(str).str.extract(r"^(\d+):(\d+)$").astype(float)
    fin_secs = mmss[0] * 60 + mmss[1]
    r["dur"] = np.clip(np.where(r["round"] < r["finish_round"], 300.0, fin_secs), 0, 300)
    # A sub-minute round gives a wildly unstable rate: a 10-strike flurry in
    # 15s reads as 40/min and would fake escalation. 60-second floor.
    r = r[r["dur"] >= 60].copy()
    r["out_pm"] = r["sig_str_att"] / (r["dur"] / 60.0)
    return r


def cardio_profile(r, f):
    """Output rate by round for 3-round fights, split winner/loser and
    decision/finish. Pooling winner+loser would hide the very thing being
    tested: a man who gasses should fade while the man finishing him climbs."""
    d = r[r["time_format"].astype(str).str.contains("3 Rnd", na=False)].copy()
    d["ended"] = np.where(d["method"].astype(str).str.contains("Decision", na=False),
                          "decision", "finish")
    long = pd.concat([
        pd.DataFrame({"event_name": f["event_name"], "bout": f["bout"],
                      "fighter": f[f"fighter_{s}"],
                      "side": np.where(f[f"outcome_{s}"] == "W", "won", "lost")})
        for s in ("a", "b")], ignore_index=True)
    long = long.drop_duplicates(["event_name", "bout", "fighter"])
    d = d.merge(long, on=["event_name", "bout", "fighter"], how="inner")
    return d.groupby(["ended", "side", "round"]).agg(
        n=("out_pm", "size"), out_per_min=("out_pm", "mean")).round(2)


# ── charts (same visual style as train.py) ────────────────────────────────────

INK, STEEL, GOLD, GREEN, GRID = "#1b2733", "#9aa7b4", "#c0891e", "#0f8a6c", "#e7eaed"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlepad": 14,
    "axes.labelsize": 11.5, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": "#c9d0d6", "axes.linewidth": 1.0,
    "xtick.color": "#5b6670", "ytick.color": "#5b6670",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
})

PRETTY = {
    "d_age_years": "Age difference (years)",
    "d_zodiac_sign_idx": "Zodiac sign difference",
    "d_name_length": "Name-length difference",
    "moon_illumination": "Moon illumination",
    "moon_age_days": "Moon age (days)",
    "is_full_moon_pm1.5d": "Full moon (±1.5d)",
    "is_new_moon_pm1.5d": "New moon (±1.5d)",
    "mercury_retrograde": "Mercury retrograde",
    "friday_the_13th": "Friday the 13th",
    "same_zodiac_sign": "Same zodiac sign",
    "same_zodiac_element": "Same zodiac element",
}


def plot_scoreboard(tab, bands, path):
    win = tab[tab["target"] == "focal_win"].sort_values("effect")
    cha = (tab[tab["target"] != "focal_win"]
           .sort_values("effect", ascending=False)
           .groupby("feature", sort=False).first()  # each feature's best shot
           .sort_values("effect").reset_index())
    fig, axes = plt.subplots(
        2, 1, figsize=(8.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [len(win), len(cha)], "hspace": 0.32})

    def panel(ax, d, band, title):
        yy = np.arange(len(d))
        ax.axvspan(0, band[0], color=GRID, zorder=0)
        ax.axvline(band[1], color="#5b6670", lw=1.2, ls=(0, (5, 4)), zorder=2)
        for y0, r in zip(yy, d.itertuples()):
            real = r.effect > band[1]
            ax.plot([0, r.effect], [y0, y0], color=GRID, lw=1.6, zorder=2)
            ax.scatter([r.effect], [y0], s=110, zorder=4,
                       color=GOLD if real else STEEL,
                       edgecolor="white", linewidth=1.4)
            ax.annotate(f"{r.effect:.3f}", (r.effect, y0), xytext=(9, 0),
                        textcoords="offset points", va="center",
                        fontsize=9.5, color=INK)
        ax.set_yticks(yy, [PRETTY.get(n, n) for n in d["feature"]])
        ax.set_title(title, loc="left", fontsize=12.5)
        ax.grid(axis="x", color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    # Conservative band for the character panel: the widest across its targets.
    cband = max((bands[t] for t in bands if t != "focal_win"), key=lambda b: b[1])
    panel(axes[0], win, bands["focal_win"], "Predicting the winner")
    panel(axes[1], cha, cband, "Predicting how the fight ends (each feature's best target)")
    axes[1].set_xlabel("Effect size  |AUC − 0.5|   (gray = 95% of pure noise, "
                       "dashed = best of 200 noise columns)")
    fig.suptitle("Astrology vs. the actuarial table", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def plot_cardio(prof, path):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    series = [("finish", "won", GREEN, "-", "Finish — winner"),
              ("decision", "won", GREEN, (0, (5, 3)), "Decision — winner"),
              ("finish", "lost", STEEL, "-", "Finish — loser"),
              ("decision", "lost", STEEL, (0, (5, 3)), "Decision — loser")]
    for ended, side, color, ls, label in series:
        s = prof.loc[(ended, side)]["out_per_min"].reindex([1, 2, 3])
        ax.plot(s.index, s.values, color=color, ls=ls, lw=2.4, zorder=3)
        ax.scatter(s.index, s.values, s=68, color=color, edgecolor="white",
                   linewidth=1.4, zorder=4)
        ax.annotate(f"{label}  ({s.values[-1]:.1f})", (3, s.values[-1]),
                    xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_xticks([1, 2, 3], ["Round 1", "Round 2", "Round 3"])
    ax.set_xlim(0.85, 3.9)
    ax.set_ylabel("Significant-strike attempts per minute")
    ax.set_title("Nobody gasses — losers are slower from round 1", loc="left")
    ax.grid(color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# ── run ───────────────────────────────────────────────────────────────────────

def main():
    f = load()
    tab, bands = cosmic_tables(f)
    print("\n=== Cosmic scoreboard (sorted by effect size) ===")
    print(tab[["feature", "target", "n", "auc", "r", "effect",
               "noise_p95", "noise_max200", "verdict"]]
          .round(3).to_string(index=False))

    print("\n=== Gate survivors, age-controlled ===")
    print(age_controlled(f).round(4).to_string(index=False))

    prof = cardio_profile(cardio_rounds(f), f)
    print("\n=== Output per minute by round, 3-round fights ===")
    print(prof.to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    plot_scoreboard(tab, bands, REPORTS_DIR / "myth_scoreboard.png")
    plot_cardio(prof, REPORTS_DIR / "myth_cardio.png")
    print(f"\ncharts -> {REPORTS_DIR / 'myth_scoreboard.png'}, "
          f"{REPORTS_DIR / 'myth_cardio.png'}")
    return tab, prof


if __name__ == "__main__":
    # Self-checks: cusp dates, a known retrograde window, round normalisation.
    assert zodiac(pd.Timestamp("1990-08-15")) == "leo"
    assert zodiac(pd.Timestamp("1990-08-23")) == "virgo"
    assert zodiac(pd.Timestamp("1990-01-01")) == "capricorn"
    assert list(mercury_retrograde(pd.Series([pd.Timestamp("2024-08-15"),
                                              pd.Timestamp("2024-07-01")]))) == [1, 0]
    assert ELEMENT["leo"] == "fire" and ELEMENT["pisces"] == "water"
    main()
