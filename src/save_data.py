import os 
import pandas as pd


def save_data(df, main_folder_path = "Data", experiment_number = 0, filename="captured_data.csv"):
    
    # 1. create the directory if it doesn't exist
    folder_path = os.path.join(main_folder_path, f"Experiment_{experiment_number}")
    print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"📂 Created folder: {folder_path}")
        
    # 2. save the dataframe to a CSV file in the directory
    
        
        