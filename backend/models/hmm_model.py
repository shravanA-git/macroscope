# backend/models/hmm_model.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


class MacroHMM:
    """
    GaussianHMM wrapper for macroeconomic regime detection.
    Uses 'full' covariance to capture correlations between indicators.
    n_states=4 maps to: Expansion, Late-Cycle, Contraction, Recovery.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 200):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
        )
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> "MacroHMM":
        self.feature_names = list(features.columns)
        self.model.fit(features.values)
        return self

    def decode(self, features: pd.DataFrame) -> np.ndarray:
        """Viterbi algorithm: most likely hidden state sequence."""
        _, states = self.model.decode(features.values, algorithm="viterbi")
        return states

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Forward-backward: posterior probability of each state at each step."""
        return self.model.predict_proba(features.values)

    def get_transition_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.transmat_,
            index=[f"state_{i}" for i in range(self.n_states)],
            columns=[f"state_{i}" for i in range(self.n_states)],
        )

    def save(self, path: Path | None = None) -> None:
        path = path or ARTIFACT_DIR / "hmm_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "MacroHMM":
        path = path or ARTIFACT_DIR / "hmm_model.pkl"
        return joblib.load(path)
