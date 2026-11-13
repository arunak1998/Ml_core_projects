import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROJECTS = {
    "🏏 IPL Winner Prediction": "ipl_winner_prediction",
    "🎬 Netflix Movie Recommender": "netflix_recommender",
    "📢 Ad CTR Prediction": "ad_ctr_prediction",
    "🛒 Walmart Demand Forecasting": "walmart_demand_forecasting",
}


def read_summary(project_dir):
    lines = (project_dir / "README.md").read_text(encoding="utf-8").splitlines()
    return next(line for line in lines[1:] if line.strip())


def run_pipeline(project_dir):
    log_box = st.empty()
    logs = ""
    process = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        logs += line
        log_box.code(logs, language=None)
    return process.wait() == 0


def show_metrics(output_dir):
    metrics_file = output_dir / "metrics.json"
    if not metrics_file.exists():
        st.info("No results yet. Click **Run pipeline**.")
        return
    metrics = json.loads(metrics_file.read_text())
    for column, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(name, value)


def show_tables(output_dir):
    for csv_file in sorted(output_dir.glob("*.csv")):
        st.subheader(csv_file.stem.replace("_", " ").title())
        table = pd.read_csv(csv_file)
        if "date" in table.columns:
            st.line_chart(table, x="date")
        st.dataframe(table, width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="ML Projects", page_icon="🤖", layout="wide")
    name = st.sidebar.radio("Project", list(PROJECTS))
    project_dir = ROOT / PROJECTS[name]

    st.title(name)
    st.caption(read_summary(project_dir))

    if st.button("▶ Run pipeline", type="primary"):
        if not run_pipeline(project_dir):
            st.error("Pipeline failed. Check the logs above.")

    show_metrics(project_dir / "outputs")
    show_tables(project_dir / "outputs")


main()
