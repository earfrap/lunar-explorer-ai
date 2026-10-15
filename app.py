"""
Lunar Explorer AI — Mission: Find Water
Interactive R&D demonstration.

All lunar environments and geophysical measurements
are synthetic.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle, Circle, Polygon

# --------------------------------------------------
# Project imports
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner_v3 import MissionPlannerV3
from lunar_explorer.scientific_metrics import (
    is_accessible_ice,
    is_positive_drill_measurement,
    is_validated_confirmation,
)
from lunar_explorer.spatial_metrics import (
    evaluate_spatial_confirmations,
)
from evaluate_missions import (
    evaluate_observations,
    run_systematic_mission,
)


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Lunar Explorer AI",
    page_icon="🌙",
    layout="wide",
)

st.title("🌙 Lunar Explorer AI")
st.subheader("Mission: Find Water")

st.markdown(
    """
    **AI-assisted autonomous exploration of synthetic
    lunar water-ice deposits.**

    This research prototype combines machine learning,
    simulated geophysical sensing, autonomous navigation,
    and selective drilling.
    """
)

st.info(
    "Scientific scope: all terrain, ice deposits and sensor "
    "measurements are synthetic. This is a methodological "
    "demonstration, not a validated lunar mission system."
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_predictor():

    predictor = IcePredictor()
    predictor.train()

    return predictor


# --------------------------------------------------
# Mission simulation
# --------------------------------------------------

def run_mission(seed, max_measurements, max_distance):

    env = LunarEnvironment(seed=seed).generate()

    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    predictor = load_predictor()

    planner = MissionPlannerV3(
        env,
        rover,
        predictor,
        max_slope=25.0
    )

    # Save the initial state before any measurement.
    history = [{
        "probability_map": planner.probability_map.copy(),
        "path": list(planner.path_history),
        "observations": [],
        "distance": 0.0,
    }]

    for _ in range(max_measurements):

        observation = planner.explore_step(
            drill=True,
            max_distance=max_distance,
        )

        if observation is None:
            break

        history.append({
            "probability_map": planner.probability_map.copy(),
            "path": list(planner.path_history),
            "observations": [
                obs.copy()
                for obs in rover.observations
            ],
            "distance": rover.distance_travelled,
        })

    ai_metrics = evaluate_observations(env, rover)

    # Independent systematic rover on the same terrain.
    systematic_metrics = run_systematic_mission(
        env,
        predictor,
        seed,
        max_measurements=max_measurements,
        max_distance=max_distance,
    )

    return {
        "environment": env,
        "history": history,
        "ai_metrics": ai_metrics,
        "systematic_metrics": systematic_metrics,
    }


def add_rover_icon(ax, x, y):
    """
    Draw a small stylized lunar rover at (x, y).
    Icon dimensions are in display pixels.
    """

    drawing = DrawingArea(34, 27, 0, 0)

    # Wheels
    for wheel_x in (7, 25):
        drawing.add_artist(
            Circle(
                (wheel_x, 5),
                radius=4,
                facecolor="#333333",
                edgecolor="white",
                linewidth=0.8,
            )
        )

    # Main rover body
    drawing.add_artist(
        Rectangle(
            (6, 9),
            22,
            10,
            facecolor="#E8E8E8",
            edgecolor="#333333",
            linewidth=1.2,
        )
    )

    # Solar panel
    drawing.add_artist(
        Rectangle(
            (10, 20),
            15,
            4,
            facecolor="#2471A3",
            edgecolor="white",
            linewidth=0.6,
        )
    )

    # Camera mast
    drawing.add_artist(
        Rectangle(
            (16, 19),
            2,
            6,
            facecolor="#555555",
        )
    )

    # Camera
    drawing.add_artist(
        Circle(
            (17, 25),
            radius=2,
            facecolor="#F4D03F",
            edgecolor="#333333",
        )
    )

    annotation = AnnotationBbox(
        drawing,
        (x, y),
        frameon=False,
        box_alignment=(0.5, 0.25),
        zorder=10,
    )

    ax.add_artist(annotation)



def build_drilling_analysis(env, observations):
    """
    Build a post-mission drilling analysis.

    Ground-truth information is used exclusively
    for evaluation, never for rover decision-making.
    """

    rows = []
    drilled_positions = []

    for step, obs in enumerate(observations, start=1):

        if not obs.get("drill_attempted", False):
            continue

        row = int(obs["row"])
        col = int(obs["col"])

        position = (row, col)
        drilled_positions.append(position)

        drill_value = obs.get("drill")
        sensor_positive = is_positive_drill_measurement(obs)

        ground_truth_positive = is_accessible_ice(
            env, row, col
        )

        confirmed = is_validated_confirmation(env, obs)



        rows.append(
            {
                "Step": step,
                "Position": f"({row}, {col})",
                "Predicted ice probability": round(
                    float(obs["ice_probability"]), 3
                ),
                "Drill measurement": (
                    round(float(drill_value), 3)
                    if drill_value is not None
                    else None
                ),
                "Ground truth ice": round(
                    float(env.ice_content[row, col]), 3
                ),
                "Ice depth (m)": round(
                    float(env.ice_depth[row, col]), 3
                ),
                "Sensor result": (
                    "Positive"
                    if sensor_positive
                    else "Negative"
                ),

                "Ground truth": (
                    "Accessible ice"
                    if ground_truth_positive
                    else "No accessible ice"
                ),

                "Result": (
                    "Confirmed"
                    if confirmed
                    else (
                        "False positive"
                        if sensor_positive
                        else (
                            "Missed accessible ice"
                            if ground_truth_positive
                            else "Not confirmed"
                        )
                    )
                ),
            }
        )

    unique_positions = len(set(drilled_positions))
    repeated_drills = (
        len(drilled_positions) - unique_positions
    )

    return rows, unique_positions, repeated_drills

# --------------------------------------------------
# Map plotting
# --------------------------------------------------

def plot_map(
    data,
    title,
    extent,
    cmap="viridis",
    vmin=None,
    vmax=None,
    path=None,
    observations=None,
    show_drills=False,
    env=None,
):

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        data,
        origin="lower",
        extent=[0, extent, 0, extent],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    resolution = extent / data.shape[0]

    if path:

        path_array = np.array(path)

        x = (path_array[:, 1] + 0.5) * resolution
        y = (path_array[:, 0] + 0.5) * resolution

        ax.plot(
            x,
            y,
            color="white",
            linewidth=1.5,
            alpha=0.9,
            label="Rover path",
        )
        add_rover_icon(
            ax,
            x[-1],
            y[-1],
        )


    if observations:

        for obs in observations:

            x = (obs["col"] + 0.5) * resolution
            y = (obs["row"] + 0.5) * resolution

            drilled = obs.get("drill_attempted", False)

            positive_measurement = is_positive_drill_measurement(obs)
            confirmed = (
                is_validated_confirmation(env, obs)
                if env is not None
                else False
            )

            if confirmed:

                # Positive synthetic drill measurement.
                ax.scatter(
                    x,
                    y,
                    marker="*",
                    s=150,
                    color="#4FC3F7",
                    edgecolors="white",
                    linewidths=1,
                    zorder=8,
                )
            elif positive_measurement:

                ax.scatter(
                    x,
                    y,
                    marker="D",
                    s=65,
                    color="#E67E22",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=8,
                )

            elif drilled:
                # Drilling location.
                ax.scatter(
                    x,
                    y,
                    marker="D",
                    s=55,
                    color="#F4D03F",
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=7,
                )

            else:

                # Non-invasive measurement.
                ax.scatter(
                    x,
                    y,
                    marker="o",
                    s=22,
                    color="white",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=6,
                )

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    fig.tight_layout()

    return fig


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Mission Configuration")

seed = st.sidebar.number_input(
    "Environment seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

max_measurements = st.sidebar.slider(
    "Maximum measurements",
    min_value=5,
    max_value=40,
    value=20,
)

max_distance = st.sidebar.slider(
    "Travel budget (m)",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100,
)

st.sidebar.markdown(
    "**Selective drilling threshold:** 0.60"
)

run_button = st.sidebar.button(
    "🚀 Run Mission",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Run and store mission
# --------------------------------------------------

if run_button:
    st.session_state["mission_step"] = 0
    with st.spinner("Running lunar exploration..."):

        st.session_state["mission"] = run_mission(
            seed=int(seed),
            max_measurements=max_measurements,
            max_distance=max_distance,
        )

        st.session_state["mission_config"] = {
            "seed": int(seed),
            "measurements": max_measurements,
            "distance": max_distance,
        }


if "mission" not in st.session_state:

    st.markdown(
        """
        ### Ready for exploration

        Configure the mission in the sidebar and select
        **Run Mission** to launch the rover simulation.
        """
    )

    st.stop()


mission = st.session_state["mission"]

env = mission["environment"]
history = mission["history"]

ai = mission["ai_metrics"]
systematic = mission["systematic_metrics"]



# --------------------------------------------------
# Main navigation
# --------------------------------------------------

tab_environment, tab_exploration, tab_performance = st.tabs([
    "🌑 Lunar Environment",
    "🤖 Autonomous Exploration",
    "📊 Mission Performance",
])


# --------------------------------------------------
# TAB 1 — Lunar Environment
# --------------------------------------------------

with tab_environment:

    st.header("Synthetic Lunar Environment")

    st.markdown(
        """
        A synthetic lunar south-pole-like environment
        containing terrain, temperature variations
        and hidden subsurface ice deposits.

        The ice map is **ground truth used for evaluation**.
        It is not available to the rover's navigation system.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = plot_map(
            env.topography,
            "Synthetic Topography",
            env.extent,
            cmap="terrain",
        )

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        fig = plot_map(
            env.temperature,
            "Surface Temperature (K)",
            env.extent,
            cmap="coolwarm",
        )

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Hidden Water-Ice Distribution")

    fig = plot_map(
        env.ice_content,
        "Synthetic Ice Content — Ground Truth",
        env.extent,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# TAB 2 — Autonomous Exploration
# --------------------------------------------------

with tab_exploration:

    st.header("AI-Guided Autonomous Exploration")

    st.markdown(
        """
        The rover selects targets using its estimated
        ice probability map.

        After each visit, synthetic neutron and GPR
        measurements update the local probability map.

        Drilling is performed only when the
        sensor-informed probability is at least 0.60.
        """
    )


    st.markdown("### 👆 Mission Timeline")

    st.markdown(
        "Move through the mission to see how the rover "
        "explores, updates its predictions and decides "
        "when to drill."
    )

    max_step = len(history) - 1

    if "mission_step" not in st.session_state:
        st.session_state["mission_step"] = 0

    # Reset the timeline when a new mission is shorter.
    if st.session_state["mission_step"] > max_step:
        st.session_state["mission_step"] = 0

    previous_col, progress_col, next_col = st.columns(
        [1, 3, 1]
    )

    with previous_col:

        if st.button(
            "◀ Previous",
            disabled=st.session_state["mission_step"] == 0,
            use_container_width=True,
        ):
            st.session_state["mission_step"] -= 1

    with next_col:

        if st.button(
            "Next ▶",
            disabled=(
                st.session_state["mission_step"] >= max_step
            ),
            use_container_width=True,
        ):
            st.session_state["mission_step"] += 1

    with progress_col:

        st.slider(
            "Exploration step",
            min_value=0,
            max_value=max_step,
            key="mission_step",
        )

    step = st.session_state["mission_step"]

    st.caption(
        f"MISSION PROGRESS — Step {step} of {max_step}"
    )

    st.progress(
        step / max_step if max_step else 0.0
    )

    state = history[step]

    col1, col2 = st.columns(2)

    with col1:

        fig = plot_map(
            state["probability_map"],
            f"Ice Probability — Step {step}",
            env.extent,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        fig = plot_map(
            state["probability_map"],
            "Rover Trajectory and Measurements",
            env.extent,
            cmap="viridis",
            vmin=0,
            vmax=1,
            path=state["path"],
            observations=state["observations"],
            show_drills=True,
            env=env,
        )

        st.pyplot(fig)
        plt.close(fig)

    observations = state["observations"]
    # --------------------------------------------------
    # Mission event feedback
    # --------------------------------------------------

    if observations:

        latest = observations[-1]

        drilled = latest.get(
            "drill_attempted",
            False
        )

        drill_value = latest.get("drill")

        ice_confirmed = is_validated_confirmation(
            env, latest
        )

        if ice_confirmed:

            st.success(
                "🏆 WATER ICE CONFIRMED! "
                f"Synthetic drill measurement: "
                f"{drill_value:.3f}"
            )

            # Celebrate only when entering a new
            # successful exploration step.
            celebration_key = (
                mission["environment"].extent,
                st.session_state["mission_config"]["seed"],
                step,
            )

            if (
                st.session_state.get(
                    "last_celebrated_step"
                ) != celebration_key
            ):

                st.balloons()

                st.session_state[
                    "last_celebrated_step"
                ] = celebration_key
        elif is_positive_drill_measurement(latest):

            st.warning(
                "⚠️ Positive synthetic drill measurement, "
                "but accessible ice was not confirmed "
                "by ground truth."
            )

        elif drilled:

            st.warning(
                "🟡 Drilling completed — "
                "Negative drill measurement."
            )

        else:

            probability = latest[
                "ice_probability"
            ]

            st.info(
                "🔍 Geophysical measurements acquired. "
                f"Estimated ice probability: "
                f"{probability:.1%}. "
                "Drilling skipped."
            )
    drills = sum(
        obs.get("drill_attempted", False)
        for obs in observations
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Measurements",
        len(observations),
    )

    c2.metric(
        "Drill Attempts",
        drills,
    )

    c3.metric(
        "Distance Travelled",
        f"{state['distance']:.0f} m",
    )

    if observations:

        last = observations[-1]

        st.markdown("#### Latest Geophysical Observation")

        st.write({
            "Position": (
                last["row"],
                last["col"],
            ),
            "Neutron proxy": round(
                last["neutron"], 3
            ),
            "GPR proxy": round(
                last["gpr"], 3
            ),
            "Estimated ice probability": round(
                last["ice_probability"], 3
            ),
            "Drilling decision": (
                "DRILL"
                if last["drill_attempted"]
                else "SKIP"
            ),
        })
    st.caption(
        "⚪ Non-invasive measurement  |  "
        "🟡 Negative drill measurement  |  "
        "🟠 Positive drill, not validated  |  "
        "🔵 Validated ice confirmation  |  "
        "🤖 Current rover position"
    )



# --------------------------------------------------
# TAB 3 — Mission Performance
# --------------------------------------------------

with tab_performance:

    st.header("Mission Performance")

    st.markdown(
        """
        Comparison between AI-guided exploration
        and a systematic geometric exploration strategy.

        Both strategies use the same synthetic environment,
        geophysical sensors, ML-based drilling rule
        and mission budgets.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "AI Detections",
        ai["detections"],
    )

    col2.metric(
        "AI Confirmations",
        ai["confirmed"],
    )

    col3.metric(
        "AI Drill Attempts",
        ai["drill_attempts"],
    )

    col4.metric(
        "AI Distance",
        f"{ai['distance']:.0f} m",
    )

    st.subheader("AI vs Systematic Exploration")

    metric_labels = {
        "measurements": "Measurements",
        "detections": "Ground-truth detections",
        "drill_attempts": "Drill attempts",
        "drills_avoided": "Drills avoided",
        "confirmed": "Drill confirmations",
        "detection_rate": "Detection rate",
        "drill_success_rate": "Drill success rate",
        "distance": "Distance (m)",
        "detections_per_100m": "Detections / 100 m",
    }

    comparison = {
        label: {
            "Adaptive AI": ai[key],
            "Systematic": systematic[key],
        }
        for key, label in metric_labels.items()
    }

    st.dataframe(
        comparison,
        use_container_width=True,
    )

    st.markdown(
        """
        **Interpretation**

        - *Ground-truth detections* count explored
          locations containing synthetic ice.
          They are not direct sensor confirmations.
        - *Drill confirmations* require a positive
          synthetic drilling measurement and accessible
          ice according to the simulation ground truth.
        - *Drills avoided* count explored locations
          where the drilling rule was not triggered.
        - These results describe one synthetic
          environment, not the full evaluation set.
        """
    )


    st.divider()

    st.subheader("🔬 Drilling Analysis")

    st.caption(
        "Post-mission analysis of synthetic drilling results. "
        "Ground-truth ice content and depth are shown only "
        "for scientific evaluation and were not available "
        "to the rover during exploration."
    )


    drilling_rows, unique_positions, repeated_drills = (
        build_drilling_analysis(
            env,
            history[-1]["observations"],
        )
    )

    if drilling_rows:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Drill attempts",
                len(drilling_rows),
            )

        with col2:
            st.metric(
                "Unique drill locations",
                unique_positions,
            )

        with col3:
            st.metric(
                "Repeated drills",
                repeated_drills,
            )

        st.dataframe(
            drilling_rows,
            use_container_width=True,
            hide_index=True,
        )

        if repeated_drills == 0:

            st.success(
                "All drilling attempts were performed "
                "at distinct locations."
            )

        else:

            st.warning(
                f"{repeated_drills} drilling attempts "
                "were performed at previously drilled "
                "locations."
            )

    else:

        st.info(
            "No drilling attempts were performed "
            "during this mission."
        )

    # --------------------------------------------------
    # Spatial analysis of validated ice confirmations
    # --------------------------------------------------

    st.divider()

    st.subheader("Spatial Distribution of Confirmed Ice")

    st.caption(
        "Post-hoc analysis using synthetic ground truth. "
        "Spatial groups are defined by a distance threshold "
        "and do not necessarily represent independent "
        "geological deposits."
    )

    spatial_results = {}

    final_observations = mission["history"][-1]["observations"]

    for threshold in (15.0, 25.0, 50.0):

        spatial_results[threshold] = (
            evaluate_spatial_confirmations(
                observations=final_observations,
                environment=env,
                distance_threshold=threshold,
            )
        )

    reference = spatial_results[25.0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Validated confirmation sites",
            reference["total_confirmed"],
        )

    with col2:
        st.metric(
            "Maximum site separation",
            f'{reference["maximum_separation_m"]:.1f} m',
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Spatial groups — 15 m",
            spatial_results[15.0]["spatial_groups"],
        )

    with col2:
        st.metric(
            "Spatial groups — 25 m",
            spatial_results[25.0]["spatial_groups"],
        )

    with col3:
        st.metric(
            "Spatial groups — 50 m",
            spatial_results[50.0]["spatial_groups"],
        )



# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Lunar Explorer AI | Independent R&D portfolio project | "
    "Synthetic geophysical simulation and applied machine learning"
)
