"""
Lunar Explorer AI
-----------------
Controlled comparison of two prediction targets:

1. Ice presence: ice_content >= 0.35
2. Accessible ice: ice_content >= 0.35 and ice_depth <= 1.0

Both models use identical synthetic sensor inputs.
Evaluation uses independent environments.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.sensors import LunarSensors


ICE_THRESHOLD = 0.35
MAX_DEPTH = 1.0

TRAIN_SEEDS = range(1000, 1020)
TEST_SEEDS = range(2000, 2010)

SAMPLES_PER_ENVIRONMENT = 500
RANDOM_STATE = 42


def generate_dataset(seeds):

    rng = np.random.default_rng(RANDOM_STATE)

    X = []
    y_ice = []
    y_accessible = []

    for seed in seeds:

        env = LunarEnvironment(
            size=100,
            extent=500,
            seed=seed,
        ).generate()

        sensors = LunarSensors(
            env,
            seed=seed + 10000,
        )

        rows = rng.integers(
            0, env.size, SAMPLES_PER_ENVIRONMENT
        )

        cols = rng.integers(
            0, env.size, SAMPLES_PER_ENVIRONMENT
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
            depth = env.ice_depth[row, col]

            X.append([
                temperature,
                slope,
                neutron,
                gpr,
            ])

            y_ice.append(
                int(ice >= ICE_THRESHOLD)
            )

            y_accessible.append(
                int(
                    ice >= ICE_THRESHOLD
                    and depth <= MAX_DEPTH
                )
            )

    return (
        np.asarray(X),
        np.asarray(y_ice),
        np.asarray(y_accessible),
    )


def create_model():

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model(name, model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    # Fixed exploratory threshold.
    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(
        y, probabilities
    )

    pr_auc = average_precision_score(
        y, probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * 45)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"AP:        {pr_auc:.3f}")

    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives:  {tn}")


def main():

    print("Generating training data...")

    X_train, y_ice_train, y_access_train = (
        generate_dataset(TRAIN_SEEDS)
    )

    print("Generating independent test data...")

    X_test, y_ice_test, y_access_test = (
        generate_dataset(TEST_SEEDS)
    )

    print("\nCLASS DISTRIBUTIONS")
    print("-" * 45)

    print(
        f"Training ice-positive fraction: "
        f"{y_ice_train.mean():.3f}"
    )

    print(
        f"Training accessible fraction: "
        f"{y_access_train.mean():.3f}"
    )

    print(
        f"Test ice-positive fraction: "
        f"{y_ice_test.mean():.3f}"
    )

    print(
        f"Test accessible fraction: "
        f"{y_access_test.mean():.3f}"
    )

    model_ice = create_model()
    model_accessible = create_model()

    print("\nTraining models...")

    model_ice.fit(X_train, y_ice_train)

    model_accessible.fit(
        X_train, y_access_train
    )

    print("\n" + "=" * 60)
    print("EVALUATION: ACCESSIBLE ICE")
    print("=" * 60)

    # Both models evaluated against the SAME target.
    evaluate_model(
        "Original model — ice presence target",
        model_ice,
        X_test,
        y_access_test,
    )

    evaluate_model(
        "Experimental model — accessible ice target",
        model_accessible,
        X_test,
        y_access_test,
    )

    print("\n" + "=" * 60)
    print("EVALUATION: ICE PRESENCE")
    print("=" * 60)

    evaluate_model(
        "Original model — ice presence target",
        model_ice,
        X_test,
        y_ice_test,
    )


if __name__ == "__main__":
    main()
