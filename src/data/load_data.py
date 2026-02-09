"""
Data loading and initial preprocessing module
"""
import pandas as pd
from pathlib import Path


def load_raw_data(filepath):
    """
    Load raw e-commerce data
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Raw dataframe
    """
    data = pd.read_csv(filepath)
    print(f"✅ Loaded data: {data.shape}")
    return data


def filter_usa(df):
    """
    Filter data for USA only and drop Country column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    usa_df = df[df['country'] == 'USA'].drop('country', axis=1).copy()
    print(f"✅ Filtered to USA: {usa_df.shape}")
    return usa_df