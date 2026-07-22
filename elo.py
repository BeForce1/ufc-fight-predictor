"""
ELO Rating System (with method-weighted K and Glicko-style rating deviation)
============================================================================
Computes fighter ratings from the full fight history, then exposes a lookup
so feature DataFrames can be enriched with `focal_elo`, `opp_elo`, `elo_diff`,
`focal_rd`, `opp_rd`, and `rd_diff` columns.

The ELO "model" prediction:
    P(focal wins) = 1 / (1 + 10^(-(focal_elo - opp_elo) / scale))

Upgrades over a plain ELO:
- **Method-of-victory multiplier on K**: finishes update ratings more than
  decisions; split decisions update less.
- **Glicko-style rating deviation (RD)**: each fighter carries a RD that starts
  high on debut and decays with each fight. `rd_diff` is exposed as a feature,
  which downstream models can use to discount ratings that are still noisy
  (e.g. 0- to 2-fight records).

Usage:
    from elo import EloRatings
    elo = EloRatings(K=32)
    elo.fit(fights_df)                       # requires method + weight_class is OK; both optional
    feature_df = elo.add_features(feature_df)   # adds elo_diff, rd_diff
"""

import bisect
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_ELO  = 1500.0
ELO_SCALE    = 400.0       # standard ELO scale parameter
DEFAULT_RD   = 350.0       # debut rating deviation
MIN_RD       = 50.0        # floor on RD
RD_DECAY     = 0.92        # RD multiplier per fight (rough Glicko equivalent)

# Layoff-based rating decay: fighters mean-revert toward DEFAULT_ELO during
# long inactivity. A soft threshold (LAYOFF_FREE_DAYS) prevents routine
# inter-bout spacing from eroding ratings of active fighters — only layoffs
# beyond ~9 months (the kind that signal injury/retirement/ring rust) decay.
# With LAYOFF_FREE_DAYS=270 and half-life 1095 (3yr), a fighter idle 3.75yr
# loses half their distance from neutral; routine 6-month gaps do nothing.
RATING_HALF_LIFE_DAYS = 1095.0
LAYOFF_FREE_DAYS      = 270.0
MIN_EVENT_DATE = pd.Timestamp("2010-01-01")  # pre-2010 UFC was a very different sport


def _mov_multiplier(method: str | None) -> float:
    """Method-of-victory K-multiplier. Decisions = 1.0 baseline."""
    if not method:
        return 1.0
    m = str(method).upper()
    if "SPLIT" in m and "DEC" in m:
        return 0.85
    if "KO" in m or "TKO" in m:
        return 1.15
    if "SUB" in m:
        return 1.25
    return 1.0


class EloRatings:
    """
    ELO rating system with MOV weighting and Glicko-style RD.

    Parameters
    ----------
    K : float
        Base update factor, scaled per-fight by method-of-victory multiplier.
    initial : float
        Starting ELO for every fighter on debut.
    """

    def __init__(
        self,
        K: float = 32.0,
        initial: float = DEFAULT_ELO,
        rating_half_life_days: float = RATING_HALF_LIFE_DAYS,
        min_event_date: pd.Timestamp | None = MIN_EVENT_DATE,
    ):
        self.K       = K
        self.initial = initial
        self.rating_half_life_days = rating_half_life_days
        self.min_event_date = min_event_date
        # fighter -> (list[timestamp_ns], list[elo_before], list[rd_before])
        self._history: dict[str, tuple[list, list, list]] = {}
        self._current: dict[str, float] = {}
        self._current_rd: dict[str, float] = {}
        self._last_date: dict[str, pd.Timestamp] = {}

    # ── Layoff decay helper ──────────────────────────────────────────────────

    def _apply_layoff_decay(self, fighter: str, date: pd.Timestamp) -> None:
        """
        Mean-revert rating toward `initial` based on time since last bout.
        Subtracts LAYOFF_FREE_DAYS so routine 6-month gaps don't decay anyone —
        only layoffs beyond ~9 months start moving the needle.
        """
        if self.rating_half_life_days <= 0:
            return
        last = self._last_date.get(fighter)
        if last is None:
            return
        days_idle = (date - last).days - LAYOFF_FREE_DAYS
        if days_idle <= 0:
            return
        factor = 0.5 ** (days_idle / self.rating_half_life_days)
        cur = self._current.get(fighter, self.initial)
        self._current[fighter] = self.initial + (cur - self.initial) * factor

    # ── Build ratings ─────────────────────────────────────────────────────────

    def fit(self, fights_df: pd.DataFrame) -> "EloRatings":
        """
        Process fights chronologically to build rating history.

        fights_df must have columns:
            event_date, fighter_a, fighter_b, outcome_a
        Optional columns consumed when present:
            method (for MOV-weighted K)
        """
        cols = ["event_date", "fighter_a", "fighter_b", "outcome_a"]
        if "method" in fights_df.columns:
            cols.append("method")
        df = fights_df[cols].copy()
        df["event_date"] = pd.to_datetime(df["event_date"])
        if self.min_event_date is not None:
            before = len(df)
            df = df[df["event_date"] >= self.min_event_date].copy()
            if before != len(df):
                log.info(f"ELO: dropped {before - len(df):,} pre-{self.min_event_date.date()} fights")
        df = df.sort_values("event_date").reset_index(drop=True)

        has_method = "method" in df.columns

        for _, row in df.iterrows():
            a, b = str(row["fighter_a"]), str(row["fighter_b"])
            date = row["event_date"]
            ts   = date.value  # int64 nanoseconds for fast bisect

            # Mean-revert inactive fighters toward initial before reading their rating
            self._apply_layoff_decay(a, date)
            self._apply_layoff_decay(b, date)

            elo_a = self._current.get(a, self.initial)
            elo_b = self._current.get(b, self.initial)
            rd_a  = self._current_rd.get(a, DEFAULT_RD)
            rd_b  = self._current_rd.get(b, DEFAULT_RD)

            # Record state BEFORE this fight
            self._record(a, ts, elo_a, rd_a)
            self._record(b, ts, elo_b, rd_b)

            # Expected scores
            ea = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))
            eb = 1.0 - ea

            # Actual scores
            outcome = str(row["outcome_a"]).strip().upper()
            if outcome == "W":
                sa, sb = 1.0, 0.0
            elif outcome == "L":
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            mov = _mov_multiplier(row["method"] if has_method else None)
            k_eff = self.K * mov

            # RD-aware K scaling: high RD (uncertain) fighters update faster.
            # Simple proxy: scale by RD / DEFAULT_RD, clipped to [0.5, 1.5].
            k_a = k_eff * max(0.5, min(1.5, rd_a / DEFAULT_RD))
            k_b = k_eff * max(0.5, min(1.5, rd_b / DEFAULT_RD))

            self._current[a] = elo_a + k_a * (sa - ea)
            self._current[b] = elo_b + k_b * (sb - eb)

            # RD decay per fight (Glicko-ish)
            self._current_rd[a] = max(MIN_RD, rd_a * RD_DECAY)
            self._current_rd[b] = max(MIN_RD, rd_b * RD_DECAY)

            # Record last-bout date for layoff-decay lookup
            self._last_date[a] = date
            self._last_date[b] = date

        log.info(f"ELO fit complete: {len(self._current):,} fighters rated")
        return self

    def _record(self, fighter: str, ts_ns: int, elo: float, rd: float) -> None:
        if fighter not in self._history:
            self._history[fighter] = ([], [], [])
        times, elos, rds = self._history[fighter]
        times.append(ts_ns)
        elos.append(elo)
        rds.append(rd)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_elo_before(self, fighter: str, date: pd.Timestamp) -> float:
        """
        Rating for `fighter` as of just before `date`.

        `_record()` stores the pre-bout rating at the same index as the bout
        itself, so for a date that exactly matches one of the fighter's own
        recorded bouts (the normal case — both training and serving query a
        fight's own date), `bisect_left` lands on that bout's index and
        `elos[idx]` is precisely the rating they carried into it.

        - For historical dates (during/before last recorded bout): return the
          stored before-bout rating; layoff-decay between bouts was already
          applied during fit().
        - For future dates (after last recorded bout): apply layoff-decay from
          the last bout to `date` so long-inactive fighters regress to neutral.
        """
        if fighter not in self._history:
            return self.initial
        times, elos, _ = self._history[fighter]
        ts = date.value
        idx = bisect.bisect_left(times, ts)
        if idx == 0:
            return self.initial
        if idx < len(times):
            return elos[idx]
        # Past all recorded bouts: use current (post-last-fight) rating + decay
        base = self._current.get(fighter, self.initial)
        last_ts = times[-1]
        days_idle = (ts - last_ts) / 86_400_000_000_000 - LAYOFF_FREE_DAYS
        if self.rating_half_life_days > 0 and days_idle > 0:
            factor = 0.5 ** (days_idle / self.rating_half_life_days)
            base = self.initial + (base - self.initial) * factor
        return base

    def get_rd_before(self, fighter: str, date: pd.Timestamp) -> float:
        """RD-deviation lookup, mirroring get_elo_before's indexing."""
        if fighter not in self._history:
            return DEFAULT_RD
        times, _, rds = self._history[fighter]
        ts = date.value
        idx = bisect.bisect_left(times, ts)
        if idx == 0:
            return DEFAULT_RD
        if idx < len(times):
            return rds[idx]
        # Past all recorded bouts: current post-last-fight RD (no further decay
        # model for RD between bouts, unlike ELO's layoff mean-reversion).
        return self._current_rd.get(fighter, DEFAULT_RD)

    # ── Enrich feature DataFrame ──────────────────────────────────────────────

    def add_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add focal_elo, opp_elo, elo_diff, focal_rd, opp_rd, rd_diff columns.
        Expects columns: focal_fighter, opp_fighter, event_date.
        """
        df = feature_df.copy()
        dates = pd.to_datetime(df["event_date"])

        focal = [(self.get_elo_before(str(f), d), self.get_rd_before(str(f), d))
                 for f, d in zip(df["focal_fighter"], dates)]
        opp   = [(self.get_elo_before(str(f), d), self.get_rd_before(str(f), d))
                 for f, d in zip(df["opp_fighter"], dates)]

        df["focal_elo"] = [t[0] for t in focal]
        df["opp_elo"]   = [t[0] for t in opp]
        df["elo_diff"]  = df["focal_elo"] - df["opp_elo"]
        df["focal_rd"]  = [t[1] for t in focal]
        df["opp_rd"]    = [t[1] for t in opp]
        df["rd_diff"]   = df["focal_rd"] - df["opp_rd"]
        return df

    # ── Direct probability prediction ─────────────────────────────────────────

    def predict_proba_from_diff(
        self, elo_diff: np.ndarray, scale: float = ELO_SCALE
    ) -> np.ndarray:
        """P(focal wins) = 1 / (1 + 10^(-elo_diff / scale))."""
        return 1.0 / (1.0 + np.power(10.0, -elo_diff / scale))

    # ── Grid search over K ────────────────────────────────────────────────────

    @classmethod
    def tune_K(
        cls,
        fights_df: pd.DataFrame,
        feature_train: pd.DataFrame,
        feature_val: pd.DataFrame,
        k_values: list[float] | None = None,
    ) -> "EloRatings":
        from sklearn.metrics import roc_auc_score

        if k_values is None:
            k_values = [16.0, 24.0, 32.0, 48.0, 64.0]

        best_auc, best_model = -1.0, None
        for K in k_values:
            model = cls(K=K).fit(fights_df)
            val   = model.add_features(feature_val)
            valid = val["elo_diff"].notna() & val["focal_win"].notna()
            if valid.sum() < 10:
                continue
            proba = model.predict_proba_from_diff(val.loc[valid, "elo_diff"].values)
            auc   = roc_auc_score(val.loc[valid, "focal_win"].astype(int), proba)
            if auc > best_auc:
                best_auc, best_model = auc, model
        log.info(f"  ELO best K={best_model.K}  val_AUC={best_auc:.4f}")
        return best_model
