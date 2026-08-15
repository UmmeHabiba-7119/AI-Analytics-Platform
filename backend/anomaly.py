import pandas as pd


def detect_sales_anomalies(df: pd.DataFrame):
    """
    Detect anomalies in Total_Sales using the IQR method.

    Returns:
        anomaly_df
        lower_bound
        upper_bound
    """

    if "Total_Sales" not in df.columns:
        return pd.DataFrame(), None, None

    working_df = df.copy()

    sales = working_df["Total_Sales"].dropna()

    if sales.empty:
        return pd.DataFrame(), None, None

    q1 = sales.quantile(0.25)
    q3 = sales.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    anomaly_df = working_df[
        (working_df["Total_Sales"] < lower_bound)
        |
        (working_df["Total_Sales"] > upper_bound)
    ].copy()

    anomaly_df["Anomaly_Type"] = anomaly_df[
        "Total_Sales"
    ].apply(
        lambda value:
        "Low Sales Anomaly"
        if value < lower_bound
        else "High Sales Anomaly"
    )

    anomaly_df = anomaly_df.sort_values(
        "Total_Sales",
        ascending=False
    )

    return (
        anomaly_df,
        lower_bound,
        upper_bound
    )