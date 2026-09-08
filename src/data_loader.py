import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Универсальная загрузка данных.

    Поддерживает:
    - CSV
    - Parquet

    Parameters
    ----------
    file_path : str
        Путь к файлу в папке data/

    Returns
    -------
    pd.DataFrame
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден")

    if path.suffix == ".csv":
        df = pd.read_csv(path)

    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)

    else:
        raise ValueError("Поддерживаются только .csv и .parquet")

    return df
