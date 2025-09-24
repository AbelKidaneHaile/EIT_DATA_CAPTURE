import os 
import pandas as pd


def create_folder(experiment_number = 0, class_number = 0, main_folder_path = "Data"):
    # code to check if a folder exists and create it if it doesn't
    folder_path = os.path.join(main_folder_path, f"Experiment_{experiment_number}/Class_{class_number}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"📂 Created folder: {folder_path}")

def create_excel_if_not_exists(directory, filename, sheet_name="Sheet1"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        # Create an empty DataFrame
        df = pd.DataFrame()
        # Save to Excel
        df.to_excel(file_path, index=False, sheet_name=sheet_name)
        print(f"Excel file created: {file_path}")
    else:
        print(f"Excel file already exists: {file_path}")
    
    return file_path
#-------------------------------------------------------------------------------------------------------------
def save_channels_to_excel(folder_path, df):
    """
    Save columns Channel_A, Channel_B, Channel_C from a dataframe to separate Excel files.
    Each function call adds ONE ROW to each file containing all values from that column.
    
    Parameters:
        folder_path (str): Directory to save the Excel files.
        df (pd.DataFrame): DataFrame containing columns Channel_A, Channel_B, Channel_C.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Map dataframe columns to their respective Excel filenames
    column_file_map = {
        "Channel_A": "channel_a.xlsx",
        "Channel_B": "channel_b.xlsx",
        "Channel_C": "channel_c.xlsx"
    }
    
    for column, filename in column_file_map.items():
        if column not in df.columns:
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        # Convert the entire column to a single row (transpose the column)
        column_values = df[column].tolist()  # Get all values from the column
        new_row = pd.DataFrame([column_values])  # Create single row with all values
        
        if os.path.exists(file_path):
            # Read existing data and append new row
            try:
                existing_df = pd.read_excel(file_path, header=None)  # No headers since we're storing raw data
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                updated_df = new_row
        else:
            # File doesn't exist, create it with the new row
            updated_df = new_row
        
        # Save without headers since each row represents a complete dataset
        updated_df.to_excel(file_path, index=False, header=False)