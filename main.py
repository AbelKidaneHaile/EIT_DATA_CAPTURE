import time

import numpy as np
import pandas as pd
import plotly.express as px
import serial
import streamlit as st
from stqdm import stqdm

import src

# my variables
SERIAL_PORT = "COM3"
BAUD_RATE = 2_000_000
NO_BYTES = 3712  # 512
EXCITATION_PATTERN = "shortened_opposite_side"  # or "square_wave"
MAX_PROGRESS = 24


def mysidebar():
    with st.sidebar:
        st.header("Settings")
        # if st.sidebar.button("CONNNECT", type="primary"):
        #     with st.spinner("Connecting to the device...", show_time=True):
        #         time.sleep(5) # REMOVE LATER
        #     st.success("Done!")

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
                    f"Time must not be above {16}. This could lead to overinflation."
                )
        except ValueError:
            st.sidebar.warning("Please enter a valid integer.")
            time_val = 0

        # Inflate and Deflate logic
        if st.sidebar.button("Inflate"):
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
    # progress_percent = st.session_state.progress / MAX_PROGRESS
    # st.progress(
    #     progress_percent, text=f"Progress: {st.session_state.progress} / {MAX_PROGRESS}"
    # )
    
    # no_captures = st.slider("How many separate captures?", 1, 100, 1)
    # inflation_class = st.slider("Inflation Class", 0, 16, 0)
    # no_repetitions = st.slider("Repetitions", 1, 10, 1)
    
    if st.button("Capture Data", type="primary"):
        with st.spinner("Wait for it...", show_time=True):
            df = src.read_frame_opp(
                serial_port=SERIAL_PORT,
                no_bytes=NO_BYTES,
                baud_rate=BAUD_RATE,
                timeout=1,
            )
        # if df is not None:
        #     st.write("Data captured successfully!")
        #     st.subheader("Channel A")
        #     st.line_chart(df["Channel_A"])

        #     st.subheader("Channel B")
        #     st.line_chart(df["Channel_B"])

        #     st.subheader("Channel C")
        #     st.line_chart(df["Channel_C"])
        if df is not None:
            st.write("Data captured successfully!")

            st.subheader("Channel A")
            fig_a = px.line(df, y="Channel_A", title="Channel A")
            # fig_a.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_a, use_container_width=True)

            st.subheader("Channel B")
            fig_b = px.line(df, y="Channel_B", title="Channel B")
            # fig_b.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("Channel C")
            fig_c = px.line(df, y="Channel_C", title="Channel C")
            st.plotly_chart(fig_c, use_container_width=True)
            
            # fig_a.update_layout(yaxis_range=[0, 1], width=1000)
            # fig_b.update_layout(yaxis_range=[0, 1], width=1000)
            # fig_c.update_layout(width=1000)
        else:
            st.error(
                "Failed to capture data. Please check the connection and try again."
            )


def main():
    st.set_page_config(
        page_title="Data Capture",
        page_icon="🎈",
    )
    mymainpage()
    mysidebar()


if __name__ == "__main__":
    main()
