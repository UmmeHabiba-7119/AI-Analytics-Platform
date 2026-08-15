import pandas as pd
import streamlit as st

@st.cache_data
def clean_dataset(df: pd.DataFrame):
    """
    Clean the Retail Sales dataset and rename selected
    columns to the application's internal naming convention.
    """

    cleaned_df = df.copy()

    # Remove unnecessary CSV index column
    cleaned_df = cleaned_df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    # Rename columns for application compatibility
    cleaned_df = cleaned_df.rename(
        columns={
            "Sales_ID": "Order_ID",
            "Sales_Amount": "Total_Sales",
            "Discount": "Discount_Percent",
            "Sales_Region": "Region",
            "Date_of_Sale": "Order_Date"
        }
    )

    # Convert date
    if "Order_Date" in cleaned_df.columns:
        cleaned_df["Order_Date"] = pd.to_datetime(
            cleaned_df["Order_Date"],
            errors="coerce"
        )

    # Numeric cleaning
    numeric_columns = [
        "Total_Sales",
        "Discount_Percent",
        "Customer_Age"
    ]

    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )

            # Median imputation
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

    # Text cleaning
    text_columns = cleaned_df.select_dtypes(
        include="object"
    ).columns

    for column in text_columns:
        cleaned_df[column] = (
            cleaned_df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # Remove duplicates
    cleaned_df = (
        cleaned_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return cleaned_df