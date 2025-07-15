import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
import plotly.express as px
import serial

import src

# Constants
SERIAL_PORT = "COM3"
BAUD_RATE = 2_000_000
NO_BYTES = 3712
MAX_PROGRESS = 24

# State
progress = 0


def dummy_function():
    pass  # placeholder for any future functionality


def validate_time_input(raw_input, progress):
    try:
        time_val = int(raw_input)
        if time_val < 0:
            print("[WARNING] Time must be non-negative. Resetting to 0.")
            return 0
        elif time_val > (16 - progress):
            print(f"[WARNING] Time must not be above 16. Resetting to 1.")
            return 1
        return time_val
    except ValueError:
        print("[WARNING] Invalid input. Resetting to 0.")
        return 0


def inflate(progress, time_val):
    dummy_function()
    return min(progress + time_val, MAX_PROGRESS)


def deflate(progress, time_val):
    return max(progress - time_val, 0)


def show_progress(progress):
    color = "red" if progress > 12 else "green"
    print(f"[PROGRESS] {progress}/{MAX_PROGRESS} - Status Color: {color}")


def capture_data():
    print("[INFO] Capturing data...")
    df = src.read_frame(
        serial_port=SERIAL_PORT, no_bytes=NO_BYTES, baud_rate=BAUD_RATE, timeout=1
    )
    if df is not None:
        print("[INFO] Data captured successfully!")
        return df
    else:
        print("[ERROR] Failed to capture data.")
        return None


def plot_channels(df):
    ddf = dd.from_pandas(df, npartitions=1)

    for channel in ["Channel_A", "Channel_B", "Channel_C"]:
        if channel in ddf.columns:
            fig = px.line(ddf.compute(), y=channel, title=channel)
            fig.show()
        else:
            print(f"[WARNING] {channel} not found in data.")


def main():
    global progress

    while True:
        print(
            "\nOptions:\n1. Inflate\n2. Deflate\n3. Show Progress\n4. Capture Data\n5. Exit"
        )
        choice = input("Enter your choice: ").strip()

        if choice in ["1", "2"]:
            raw_time = input("Enter time (0-16): ").strip()
            time_val = validate_time_input(raw_time, progress)

            if choice == "1":
                progress = inflate(progress, time_val)
                print(f"[INFO] Inflated. Current progress: {progress}")
            else:
                progress = deflate(progress, time_val)
                print(f"[INFO] Deflated. Current progress: {progress}")

        elif choice == "3":
            show_progress(progress)

        elif choice == "4":
            df = capture_data()
            if df is not None:
                plot_channels(df)

        elif choice == "5":
            print("[INFO] Exiting.")
            break
        else:
            print("[ERROR] Invalid option. Please try again.")


if __name__ == "__main__":
    main()
