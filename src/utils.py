def split_features_target(df, target_column: str):
    """
    Делит DataFrame на признаки (X) и целевую переменную (y).

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str

    Returns
    -------
    X, y
    """

    if target_column not in df.columns:
        raise ValueError(f"Колонка {target_column} не найдена")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
