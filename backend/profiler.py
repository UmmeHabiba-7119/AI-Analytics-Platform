import pandas as pd
import streamlit as st

@st.cache_data
def get_basic_profile(df: pd.DataFrame):
    """
    Generate basic data quality statistics.
    """

    total_rows = len(df)
    total_columns = len(df.columns)
    duplicate_rows = df.duplicated().sum()
    total_missing = df.isnull().sum().sum()

    profile = {
        "Total Rows": total_rows,
        "Total Columns": total_columns,
        "Duplicate Rows": duplicate_rows,
        "Total Missing Values": total_missing
    }

    return profile

@st.cache_data
def get_missing_summary(df: pd.DataFrame):
    """
    Calculate missing value count and percentage for each column.
    """

    missing_count = df.isnull().sum()

    missing_percent = (
        missing_count / len(df) * 100
    ).round(2)

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing Percentage": missing_percent.values
    })

    return summary

@st.cache_data
def get_outlier_summary(df: pd.DataFrame):
    """
    Detect outliers in numeric columns using the IQR method.
    """

    results = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ]

        results.append({
            "Column": column,
            "Outlier Count": len(outliers),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2)
        })

    return pd.DataFrame(results)