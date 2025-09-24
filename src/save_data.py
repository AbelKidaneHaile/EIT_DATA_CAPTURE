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

def save_channels_to_excel(folder_path, df):
    """
    Save columns Channel_A, Channel_B, Channel_C from a dataframe to separate Excel files.
    Each new row is appended. Files are created if they don't exist.
    
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
        file_path = os.path.join(folder_path, filename)
        # Prepare the row to append
        row_df = pd.DataFrame([df[column].values])
        
        if os.path.exists(file_path):
            # Append to existing file
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # Load existing data to find next row
                try:
                    existing_df = pd.read_excel(file_path)
                    startrow = len(existing_df)
                except:
                    startrow = 0
                row_df.to_excel(writer, index=False, header=False, startrow=startrow)
        else:
            # Create new file with header
            row_df.to_excel(file_path, index=False)
    
        
        