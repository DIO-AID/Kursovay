import numpy as np
import pandas as pd


def safe_clean(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # заполняем только оставшиеся NaN (нечисловые колонки)
    df = df.fillna(0)
    return df