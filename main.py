import time

import numpy as np
import pandas as pd
import serial
import streamlit as st
from stqdm import stqdm

import src

# my variables
SERIAL_PORT = "COM3"
MAX_PROGRESS = 24


def dummy_function():
    pass  # placeholder for any future functionality


def mysidebar():
    with st.sidebar:
        st.header("Settings")
        time_input = st.sidebar.text_input("Enter time (0–16):", "1")
    # Validate input
    try:
        time_val = int(time_input)
        if time_val < 0:
            st.sidebar.warning("Time must be non-negative.")
            time_val = 0
        elif time_val > (16 - st.session_state.progress):
            time_val = 1
            st.sidebar.warning(
                f"Time must not be above {MAX_PROGRESS}. This could lead to overinflation."
            )
    except ValueError:
        st.sidebar.warning("Please enter a valid integer.")
        time_val = 0

    # Inflate and Deflate logic
    if st.sidebar.button("Inflate"):
        dummy_function()
        st.session_state.progress = min(
            st.session_state.progress + time_val, MAX_PROGRESS
        )

    if st.sidebar.button("Deflate"):
        st.session_state.progress = max(st.session_state.progress - time_val, 0)


def mymainpage():
    st.title("Data Capture")
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    # Show progress bar
    progress_percent = st.session_state.progress / MAX_PROGRESS
    st.progress(
        progress_percent, text=f"Progress: {st.session_state.progress} / {MAX_PROGRESS}"
    )

    if st.button("Capture Data", type="primary"):
        dummy_function()

        chart_data = pd.DataFrame(
            np.random.randn(20, 1),
            columns=[
                "a",
            ],
        )
        st.line_chart(chart_data)


def main():
    mymainpage()
    mysidebar()


if __name__ == "__main__":
    main()
