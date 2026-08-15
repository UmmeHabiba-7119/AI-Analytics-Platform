import pandas as pd


def get_schema(df: pd.DataFrame):

    schema = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str),

        "Missing Values": df.isnull().sum().values,

        "Unique Values": df.nunique().values

    })

    return schema