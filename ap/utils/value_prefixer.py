import pandas as pd
from typing import List

def transform(data, col_name:str) -> pd.Series:
    return data[col_name].apply(lambda x: f"{col_name}_{x}")

def inverse_transform(data, col_name) -> pd.Series:
    return data[col_name].apply(lambda x: x.replace(f"{col_name}_", ""))

def transform_all(data, exclude: List[str] = [], include: List[str] =[]) -> pd.DataFrame:
    for col in data.columns:
        if col not in exclude and (len(include) == 0 or col in include):
            data[col] = transform(data, col)
    return data

def inverse_transform_all(data, exclude: List[str] =[], include: List[str] =[]) -> pd.DataFrame:
    for col in data.columns:
        if col not in exclude and (len(include) == 0 or col in include):
            data[col] = inverse_transform(col)
    return data

