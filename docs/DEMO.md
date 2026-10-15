# Lunar Explorer AI — Interactive Demo

## Overview

The Streamlit dashboard provides an interactive interface to the synthetic lunar exploration simulator.

Users can explore generated lunar environments and examine the behavior of ML-guided autonomous rover missions.

All displayed terrain, ice properties, and geophysical measurements are synthetic.

## Local Setup

Python 3.12.15 was used during development.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will display a local URL, typically:

`http://localhost:8501`

The application configures the required `src/` and `tests/` import paths internally.

## Public Deployment

Streamlit Community Cloud is a potential hosting option.

Before deployment:

1. Push the reviewed repository to GitHub.
2. Confirm that `requirements.txt` contains all runtime dependencies.
3. Connect the GitHub repository to Streamlit Community Cloud.
4. Select `app.py` as the application entry point.
5. Test the deployed application, including mission execution and visualizations.
6. Add the public application URL to the README.

Deployment has not yet been performed or verified.

## Suggested Demonstration Video

A short demonstration could show:

1. The generated lunar environment.
2. The rover selecting and navigating toward targets.
3. Sensor observations and drilling decisions.
4. Final mission metrics and spatial results.

A 30–60 second recording would be suitable for a future project presentation or LinkedIn post.

## Scientific Disclaimer

The application is an interactive research prototype.

Its simulated observations and predictions must not be interpreted as real lunar measurements or validated operational recommendations.
