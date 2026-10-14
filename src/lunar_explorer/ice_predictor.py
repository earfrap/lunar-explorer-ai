"""
Lunar Explorer AI
-----------------
Machine-learning prediction of subsurface water ice.

Two Random Forest models:
    1. Prior model: temperature and slope.
    2. Observation model: temperature, slope,
       neutron signal and GPR signal.

Training uses independent synthetic environments.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.sensors import LunarSensors


class IcePredictor:

    def __init__(self, ice_threshold=0.35, random_state=42):

        self.ice_threshold = ice_threshold

        self.prior_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )

        self.observation_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state + 1,
            n_jobs=-1
        )

        self.trained = False

    def generate_training_data(
        self,
        n_environments=20,
        samples_per_environment=500,
        seed=1000
    ):

        rng = np.random.default_rng(seed)

        X_prior = []
        X_observation = []
        y = []

        for i in range(n_environments):

            env = LunarEnvironment(
                size=100,
                extent=500,
                seed=seed + i
            ).generate()

            sensors = LunarSensors(
                env,
                seed=seed + 10000 + i
            )

            rows = rng.integers(
                0, env.size, samples_per_environment
            )

            cols = rng.integers(
                0, env.size, samples_per_environment
            )

            for row, col in zip(rows, cols):

                temperature = env.temperature[row, col]
                slope = env.slope[row, col]

                neutron = sensors.neutron_measurement(
                    row, col
                )

                gpr = sensors.gpr_measurement(
                    row, col
                )

                ice = env.ice_content[row, col]

                X_prior.append([
                    temperature,
                    slope
                ])

                X_observation.append([
                    temperature,
                    slope,
                    neutron,
                    gpr
                ])

                y.append(
                    int(ice >= self.ice_threshold)
                )

        return (
            np.asarray(X_prior),
            np.asarray(X_observation),
            np.asarray(y)
        )

    def train(
        self,
        n_environments=20,
        samples_per_environment=500
    ):

        X_prior, X_observation, y = (
            self.generate_training_data(
                n_environments=n_environments,
                samples_per_environment=samples_per_environment
            )
        )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Training data must contain both ice classes."
            )

        self.prior_model.fit(X_prior, y)

        self.observation_model.fit(
            X_observation, y
        )

        self.trained = True

        print(
            f"Training completed: {len(y)} samples"
        )

        print(
            f"Ice-positive fraction: {y.mean():.3f}"
        )

    def predict_prior(self, temperature, slope):

        if not self.trained:
            raise RuntimeError("Train the model first.")

        features = np.array([
            [temperature, slope]
        ])

        return float(
            self.prior_model.predict_proba(
                features
            )[0, 1]
        )

    def predict_observation(
        self,
        temperature,
        slope,
        neutron,
        gpr
    ):

        if not self.trained:
            raise RuntimeError("Train the model first.")

        features = np.array([[
            temperature,
            slope,
            neutron,
            gpr
        ]])

        return float(
            self.observation_model.predict_proba(
                features
            )[0, 1]
        )

    def predict_prior_map(self, environment):

        if not self.trained:
            raise RuntimeError("Train the model first.")

        features = np.column_stack([
            environment.temperature.ravel(),
            environment.slope.ravel()
        ])

        probabilities = (
            self.prior_model.predict_proba(
                features
            )[:, 1]
        )

        return probabilities.reshape(
            environment.temperature.shape
        )

    def evaluate(self, seed=9999, n_samples=1000):

        X_prior, X_observation, y = (
            self.generate_training_data(
                n_environments=1,
                samples_per_environment=n_samples,
                seed=seed
            )
        )

        results = {}

        for name, model, features in [
            ("Prior", self.prior_model, X_prior),
            (
                "Observation",
                self.observation_model,
                X_observation
            )
        ]:

            probabilities = model.predict_proba(
                features
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            results[name] = {
                "accuracy": accuracy_score(
                    y, predictions
                ),
                "precision": precision_score(
                    y, predictions,
                    zero_division=0
                ),
                "recall": recall_score(
                    y, predictions,
                    zero_division=0
                ),
                "roc_auc": (
                    roc_auc_score(y, probabilities)
                    if len(np.unique(y)) == 2
                    else float("nan")
                )
            }

        return results
