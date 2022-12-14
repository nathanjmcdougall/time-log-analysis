from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from column import DT_COLS, DURATION_COL, FROM_COL, TO_COL


def _pre_blank_line(blank_lines_df: DataFrame) -> DataFrame:
    is_blank_idx = blank_lines_df.isnull().all(axis="columns")
    ((idx, *_),) = np.where(is_blank_idx)  # We only care about the first blank line
    preblank_df = blank_lines_df.iloc[:idx].copy()
    return preblank_df


def get_time_log_df(csv_path: PathLike[Any]) -> DataFrame:
    """Read-in the time log csv file as a DataFrame with sensible dtypes."""
    single_blank_line_df = pd.read_csv(csv_path, skip_blank_lines=False)
    df = _pre_blank_line(single_blank_line_df)

    df[DT_COLS] = df[DT_COLS].apply(pd.to_datetime, format=str("%Y-%m-%d %H:%M"))
    df[DURATION_COL] = df[TO_COL] - df[FROM_COL]

    df.sort_values(FROM_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
