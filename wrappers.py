"""
Model wrappers (calibration + stacking), stored in a non-__main__ module so
joblib can unpickle them regardless of how the training script was invoked.
"""

from __future__ import annotations

import numpy as np


class PlattCalibratedModel:
    """Thin wrapper: applies Platt scaling on top of any predict_proba model."""

    def __init__(self, base_model, platt_lr):
        self.base_model = base_model
        self.platt_lr   = platt_lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt_lr.predict_proba(raw)
        return cal


class IsotonicCalibratedModel:
    """Isotonic-regression calibration wrapper."""

    def __init__(self, base_model, isotonic):
        self.base_model = base_model
        self.isotonic   = isotonic

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.isotonic.predict(raw)
        cal = np.clip(cal, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - cal, cal])


class StackedModel:
    """
    Stacked meta-learner: a logistic regression over calibrated base-model
    probabilities. Exposes predict_proba(X) so it can be treated as a drop-in
    model by downstream code.
    """

    def __init__(self, meta_lr, base_models, feature_names=("LR", "RF", "XGB")):
        self.meta_lr = meta_lr
        self.base_models = base_models
        self.feature_names = tuple(feature_names)

    def _base_probs(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.meta_lr.predict_proba(self._base_probs(X))
