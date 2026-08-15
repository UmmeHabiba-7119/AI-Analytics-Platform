import time
import pandas as pd


def aggregate_query(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    operation: str = "sum"
):
    """
    Perform aggregation query and return result with execution time.
    """

    start_time = time.perf_counter()

    if operation == "sum":
        result = (
            df.groupby(group_column)[value_column]
            .sum()
            .reset_index()
        )

    elif operation == "mean":
        result = (
            df.groupby(group_column)[value_column]
            .mean()
            .reset_index()
        )

    elif operation == "count":
        result = (
            df.groupby(group_column)[value_column]
            .count()
            .reset_index()
        )

    elif operation == "min":
        result = (
            df.groupby(group_column)[value_column]
            .min()
            .reset_index()
        )

    elif operation == "max":
        result = (
            df.groupby(group_column)[value_column]
            .max()
            .reset_index()
        )

    else:
        raise ValueError("Unsupported aggregation operation.")

    execution_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return result, execution_time_ms


def filtered_query(
    df: pd.DataFrame,
    filter_column: str,
    filter_value,
    group_column: str,
    value_column: str,
    operation: str = "sum"
):
    """
    Filter the dataset first, then perform aggregation.
    """

    start_time = time.perf_counter()

    filtered_df = df[
        df[filter_column] == filter_value
    ]

    if operation == "sum":
        result = (
            filtered_df.groupby(group_column)[value_column]
            .sum()
            .reset_index()
        )

    elif operation == "mean":
        result = (
            filtered_df.groupby(group_column)[value_column]
            .mean()
            .reset_index()
        )

    elif operation == "count":
        result = (
            filtered_df.groupby(group_column)[value_column]
            .count()
            .reset_index()
        )

    elif operation == "min":
        result = (
            filtered_df.groupby(group_column)[value_column]
            .min()
            .reset_index()
        )

    elif operation == "max":
        result = (
            filtered_df.groupby(group_column)[value_column]
            .max()
            .reset_index()
        )

    else:
        raise ValueError("Unsupported aggregation operation.")

    execution_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return result, execution_time_ms