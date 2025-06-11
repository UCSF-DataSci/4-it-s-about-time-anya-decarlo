import pandas as pd
import glob
import os

def load_data(data_dir):
    """
    Load and preprocess physiological data from CSV files.
    
    Args:
        data_dir (str): Path to directory containing raw data files
        
    Returns:
        pd.DataFrame: Processed data with required columns
    """
    # Get all CSV files in the directory
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # Read and combine all files
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session'])
    
    # Combine all dataframes
    data = pd.concat(dfs, ignore_index=True)
    
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Ensure all required columns exist
    required_columns = ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
    for col in required_columns:
        if col not in data.columns:
            data[col] = None
    
    return data 