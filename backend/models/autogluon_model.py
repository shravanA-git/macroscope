# backend/models/autogluon_model.py
import pandas as pd
from pathlib import Path

AUTOGLUON_DIR = Path(__file__).parent / "artifacts" / "autogluon_regime"


class RegimeTransitionPredictor:
    """
    AutoGluon TabularPredictor for regime transition forecasting.
    Discriminative counterpart to the HMM generative model.
    """

    def __init__(self, lookahead: int = 1):
        self.lookahead = lookahead
        self.predictor = None

    def prepare_training_data(
        self,
        features: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        """
        Supervised dataset: features at t → regime at t+lookahead.
        The shift creates the prediction target without look-ahead bias.
        Infinities are replaced with NaN and forward-filled to avoid tree failures.
        """
        df = features.copy()
        # Replace ±inf with NaN then forward/back fill so tree models don't fail
        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df = df.ffill().bfill()
        df["next_regime"] = regime_labels.shift(-self.lookahead)
        return df.dropna(subset=["next_regime"])

    def fit(
        self,
        train_df: pd.DataFrame,
        time_limit: int = 120,
        path: Path = AUTOGLUON_DIR,
    ) -> "RegimeTransitionPredictor":
        from autogluon.tabular import TabularPredictor

        self.predictor = TabularPredictor(
            label="next_regime",
            eval_metric="accuracy",
            path=str(path),
            verbosity=0,
        )
        self.predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            presets="medium_quality",
            excluded_model_types=["CAT", "NN_TORCH"],
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return self.predictor.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict_proba(features)

    def leaderboard(self) -> pd.DataFrame:
        return self.predictor.leaderboard(silent=True)
