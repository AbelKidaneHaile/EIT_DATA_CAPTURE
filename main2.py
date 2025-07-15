import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

import src  # Your custom module

# Constants
SERIAL_PORT = "COM3"
BAUD_RATE = 2_000_000
NO_BYTES = 3712
EXCITATION_PATTERN = "shortened_opposite_side"
MAX_PROGRESS = 24


class RealTimeCaptureApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Data Capture")
        self.progress = 0
        self.df = None  # Store captured dataframe

        # Layouts
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Time Input
        self.time_input = QtWidgets.QSpinBox()
        self.time_input.setRange(0, 16)
        self.time_input.setValue(1)

        # Buttons
        self.capture_button = QtWidgets.QPushButton("Capture Data")
        self.capture_button.clicked.connect(self.capture_data)

        self.inflate_button = QtWidgets.QPushButton("Inflate")
        self.inflate_button.clicked.connect(self.inflate)

        self.deflate_button = QtWidgets.QPushButton("Deflate")
        self.deflate_button.clicked.connect(self.deflate)

        # Progress Label
        self.progress_label = QtWidgets.QLabel(
            f"Progress: {self.progress} / {MAX_PROGRESS}"
        )

        # Layout arrangements
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel("Time (0–16):"))
        hlayout.addWidget(self.time_input)
        hlayout.addWidget(self.inflate_button)
        hlayout.addWidget(self.deflate_button)

        layout.addWidget(self.capture_button)
        layout.addLayout(hlayout)
        layout.addWidget(self.progress_label)

        # Plot Widgets
        self.plot_widgets = []
        self.curves = []
        for label in ["A", "B", "C"]:
            pw = pg.PlotWidget(title=f"Channel {label}")
            layout.addWidget(pw)
            self.plot_widgets.append(pw)
            self.curves.append(pw.plot(pen=pg.mkPen("y", width=1)))

        self.setCentralWidget(central_widget)

    def inflate(self):
        val = self.time_input.value()
        if val + self.progress > MAX_PROGRESS:
            QtWidgets.QMessageBox.warning(
                self, "Warning", f"Time must not exceed {MAX_PROGRESS}."
            )
            return
        self.progress = min(self.progress + val, MAX_PROGRESS)
        self.progress_label.setText(f"Progress: {self.progress} / {MAX_PROGRESS}")

    def deflate(self):
        val = self.time_input.value()
        self.progress = max(self.progress - val, 0)
        self.progress_label.setText(f"Progress: {self.progress} / {MAX_PROGRESS}")

    def capture_data(self):
        try:
            self.df = src.read_frame_sq(
                serial_port=SERIAL_PORT,
                no_bytes=NO_BYTES,
                baud_rate=BAUD_RATE,
                timeout=1,
            )
            if self.df is None or self.df.empty:
                raise ValueError("Empty DataFrame")

            # QtWidgets.QMessageBox.information(self, "Success", "Data captured successfully!")

            self.plot_channel(0, self.df["Channel_A"])
            self.plot_channel(1, self.df["Channel_B"])
            self.plot_channel(2, self.df["Channel_C"])

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to capture data:\n{e}"
            )

    def plot_channel(self, index, data_series):
        x = list(range(len(data_series)))
        self.curves[index].setData(x, data_series.to_numpy())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = RealTimeCaptureApp()
    win.show()
    sys.exit(app.exec_())
