import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

from columns import DT_COLS, DURATION_COL, FROM_COL, TO_COL


def _get_csv_path(env_name: str = "TIMELOG_REPORT_CSV_PATH") -> Path:
    load_dotenv()
    path_str = os.getenv(env_name)
    if path_str is None:
        raise KeyError(f"{env_name} is not set")
    path = Path(path_str)
    return path


def _pre_blank_line(blank_lines_df: DataFrame) -> DataFrame:
    is_blank_idx = blank_lines_df.isnull().all(axis="columns")
    ((idx, *_),) = np.where(is_blank_idx)  # We only care about the first blank line
    preblank_df = blank_lines_df.iloc[:idx].copy()
    return preblank_df


def get_time_log_df() -> DataFrame:
    """Read-in the time log csv file as a DataFrame with sensible dtypes."""
    csv_path = _get_csv_path()
    single_blank_line_df = pd.read_csv(csv_path)
    df = _pre_blank_line(single_blank_line_df)

    df[DT_COLS] = df[DT_COLS].apply(pd.to_datetime, format=str("%Y-%m-%d %H:%M"))
    df[DURATION_COL] = df[TO_COL] - df[FROM_COL]

    df.sort_values(FROM_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
