import numpy as np
import pandas as pd
import serial


# This function captures only one frame for shortened opposite side excitation
def read_frame_opp(serial_port, no_bytes, baud_rate=2_000_000, timeout=1):

    try:
        with serial.Serial(serial_port, baud_rate, timeout=timeout) as ser:
            ser.write(b"U")
            data = ser.read(no_bytes)
            if data:
                data = np.frombuffer(data, dtype=np.uint8)
                
                print("\n\n",len(data),data)
                # Extract Channels
                Channel_A = []
                Channel_B = []

                for i in range(0, len(data) - 3, 4):
                    val_A = (data[i] << 7) + data[i + 1]
                    val_B = (data[i + 2] << 7) + data[i + 3]
                    Channel_A.append(val_A * 0.0001220852155)
                    Channel_B.append(val_B * 0.0001220852155)

                Channel_A = np.array(Channel_A)
                Channel_B = np.array(Channel_B)

                # Convert to mA
                Channel_B = (1000 * Channel_B) / (150)

                Averages = []
                for i in range(0, 128, 4):
                    Averages.append(np.mean(Channel_B[i : i + 4]))

                # Repeat Averages to match original Channel_B length
                rep_B = np.repeat(Averages, 4)

                Current_In = 1
                correction_ratio = Current_In / Channel_B
                Channel_C = Channel_A * correction_ratio

                df = pd.DataFrame(
                    {
                        "Channel_A": Channel_A,
                        "Channel_B": Channel_B,
                        "Channel_C": Channel_C,
                    }
                )
                return df

            else:
                print("No data received.")
                return None
    except serial.SerialException as e:
        print(f"SERIAL ERROR: device not found or not connected")
        return None
