import time

import numpy as np
import pandas as pd
import serial
import streamlit as st
from stqdm import stqdm

import src

# my variables
SERIAL_PORT = "COM3"
BAUD_RATE = 2_000_000
NO_BYTES = 3712
  
MAX_PROGRESS = 24

def dummy_function():
    pass  # placeholder for any future functionality


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
    # progress_percent = st.session_state.progress / MAX_PROGRESS
    # st.progress(
    #     progress_percent, text=f"Progress: {st.session_state.progress} / {MAX_PROGRESS}"
    # )

    if st.button("Capture Data", type="primary"):
        with st.spinner("Wait for it...", show_time=True):
            df = src.read_frame(
                        serial_port=SERIAL_PORT,
                        no_bytes=NO_BYTES,
                        baud_rate=BAUD_RATE,
                        timeout=1,
                )
        if df is not None:
            st.write("Data captured successfully!")
            st.line_chart(df)
        else: 
            st.error("Failed to capture data. Please check the connection and try again.")

def main():
    st.set_page_config(
        page_title="Data Capture",
        page_icon="🎈",
    )
    mymainpage()
    mysidebar()

if __name__ == "__main__":
    main()
