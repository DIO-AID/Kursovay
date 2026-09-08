import re

def clean_feature_names(df):
    df = df.copy()

    def clean(name):
        name = str(name)
        name = re.sub(r"[^\w]+", "_", name)  # всё кроме букв/цифр/_
        name = re.sub(r"_+", "_", name)      # убрать повторные _
        return name.strip("_")

    df.columns = [clean(c) for c in df.columns]

    # защита от дублей
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate feature names after cleaning!")

    return df