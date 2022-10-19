import os
from pathlib import Path
from dotenv import load_dotenv


def get_csv_path(env_name: str = "TIMELOG_REPORT_CSV_PATH") -> Path:
    """Get the path to the time log csv file from the environment."""
    load_dotenv()
    path_str = os.getenv(env_name)
    if path_str is None:
        raise KeyError(f"{env_name} is not set")
    path = Path(path_str)
    return path
