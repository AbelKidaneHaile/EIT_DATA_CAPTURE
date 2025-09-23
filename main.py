import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import serial
import streamlit as st
from stqdm import stqdm
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

import src

# my variables
SERIAL_PORT = "COM3"
BAUD_RATE = 2_000_000
NO_BYTES = 512  # 512 # 3712 #
EXCITATION_PATTERN = "shortened_opposite_side"  # or "square_wave"
MAX_PROGRESS = 24

# streamlit customization
st.set_page_config(layout="wide")

if "arduino" not in st.session_state:
    arduino = src.ArduinoController()
    arduino.connect()
    st.session_state["arduino"] = arduino

arduino = st.session_state["arduino"]

def stream_plot(df, i, plot_placeholder):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=df["Channel_A"], mode="lines", name="Channel A")
    )  # , line=dict(color="cyan")
    # fig.add_trace(go.Scatter(y=df["Channel_B"], mode='lines', name='Channel B'))
    # fig.add_trace(go.Scatter(y=df["Channel_C"], mode='lines', name='Channel C'))

    for x in range(0, len(df), 4):
        fig.add_vline(
            x=x, line=dict(color="red", dash="dash"), opacity=0.7  # dashed red line
        )
    # fig.add_trace(go.Histogram( x=df.index,  y=df["Channel_A"],  name="Channel A", histfunc="sum",xbins=dict(size=3) ) )
    fig.update_layout(
        title=f"Frame-{i}",
        xaxis_title="Sequence Index",
        yaxis_title="Voltage",
        legend=dict(title="Channels"),
        # width=1000,
        # height=500,
    )

    plot_placeholder.plotly_chart(fig, use_container_width=True)



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
                    f"Time must not be above {16}. This could lead to overinflation."
                )
        except ValueError:
            st.sidebar.warning("Please enter a valid integer.")
            time_val = 0

        # Inflate and Deflate logic
        if st.sidebar.button("Inflate"):
            try:
                arduino.inflate(int(time_input))
                st.session_state.progress = min(
                    st.session_state.progress + time_val, MAX_PROGRESS
                )
            except Exception as e:
                st.sidebar.error(f"Error during inflation: {e}")

        if st.sidebar.button("Deflate"):
            try:
                arduino.deflate(int(time_input))
                st.session_state.progress = max(st.session_state.progress - time_val, 0)
            except Exception as e:
                st.sidebar.error(f"Error during deflation: {e}")


def mymainpage():

    st.title("Data Capture")
    if "progress" not in st.session_state:
        st.session_state.progress = 0

    if st.button("Capture Data", type="primary"):
        plot_placeholder = st.empty()
        for i in range(10000):
            with st.spinner("Wait for it...", show_time=True):
                df = src.read_frame_opp(
                    serial_port=SERIAL_PORT,
                    no_bytes=NO_BYTES,
                    baud_rate=BAUD_RATE,
                    timeout=1,
                )

                # df = src.read_frame_serial(
                #     serial_port=SERIAL_PORT,
                #     no_bytes=NO_BYTES,
                #     baud_rate=BAUD_RATE,
                #     timeout=1,
                # )

                # df = src.read_frame_sq(
                #     serial_port=SERIAL_PORT,
                #     no_bytes=NO_BYTES,
                #     baud_rate=BAUD_RATE,
                #     timeout=2,
                # )

            if df is not None:
                stream_plot(df, i, plot_placeholder)
                # st.line_chart(df)
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
