import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_dataset():
    """
    Load the Retail Sales dataset from the data folder.
    """

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "retail_sales.csv"

    return pd.read_csv(dataset_path)