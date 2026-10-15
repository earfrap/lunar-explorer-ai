"""
Shared scientific definitions for Lunar Explorer AI.

Ground truth is used exclusively for evaluation,
never for rover navigation or drilling decisions.
"""

ICE_THRESHOLD = 0.35
MAX_ACCESSIBLE_DEPTH = 1.0


def is_ground_truth_ice(env, row, col):
    """Whether synthetic ice content exceeds the threshold."""
    return bool(
        env.ice_content[row, col] >= ICE_THRESHOLD
    )


def is_accessible_ice(env, row, col):
    """Whether synthetic ice is present within drilling depth."""
    return bool(
        is_ground_truth_ice(env, row, col)
        and env.ice_depth[row, col] <= MAX_ACCESSIBLE_DEPTH
    )


def is_positive_drill_measurement(observation):
    """Whether an attempted drill produced a positive measurement."""
    value = observation.get("drill")

    return bool(
        observation.get("drill_attempted", False)
        and value is not None
        and value >= ICE_THRESHOLD
    )


def is_validated_confirmation(env, observation):
    """
    Whether a positive drill measurement agrees with
    accessible synthetic ground-truth ice.
    """
    row = int(observation["row"])
    col = int(observation["col"])

    return bool(
        is_positive_drill_measurement(observation)
        and is_accessible_ice(env, row, col)
    )
