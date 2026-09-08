import pandas as pd
import numpy as np
from utils.column_profiler import profile_dataframe, enrich_with_llm


class DataHandler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target = None
        self.task_type = None
        self.datetime_col = None
        self.id_col = None
        self.profile = None

    # =========================
    # ЗАГРУЗКА (BIG DATA SAFE)
    # =========================
    @staticmethod
    def load_file(file, nrows=None):
        if file.name.endswith(".csv"):
            return pd.read_csv(file, nrows=nrows)

        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, nrows=nrows)

        elif file.name.endswith(".parquet"):
            return pd.read_parquet(file)

        else:
            raise ValueError("Неподдерживаемый формат файла")

    # =========================
    # TYPES
    # =========================
    def detect_types(self):
        df = self.df

        numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

        datetime_cols = []
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    datetime_cols.append(col)
            except:
                pass

        self.datetime_col = datetime_cols[0] if datetime_cols else None

        return {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols,
        }

    # =========================
    # TARGET
    # =========================
    def detect_target(self):
        candidates = ["target", "price", "value", "label"]

        for col in candidates:
            if col in self.df.columns:
                self.target = col
                return col

        self.target = self.df.columns[-1]
        return self.target

    # =========================
    # ID DETECTION
    # =========================
    def detect_id_column(self):
        for col in self.df.columns:
            if "id" in col.lower():
                self.id_col = col
                return col

        self.id_col = None
        return None

    # =========================
    # TASK TYPE
    # =========================
    def detect_task_type(self):
        y = self.df[self.target]

        if y.nunique() < 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        return self.task_type

    # =========================
    # MEMORY OPTIMIZATION
    # =========================
    def optimize_memory(self):
        for col in self.df.select_dtypes(include=["float64"]).columns:
            self.df[col] = self.df[col].astype("float32")

        for col in self.df.select_dtypes(include=["int64"]).columns:
            self.df[col] = self.df[col].astype("int32")

    # =========================
    # CLEAN TARGET
    # =========================
    def clean_target(self, X, y):
        mask = y.notna()
        return X.loc[mask], y.loc[mask]

    # =========================
    # PROFILE
    # =========================
    def run_profile(self, use_llm=True):
        p = profile_dataframe(self.df)

        for col_name, col_info in p["columns"].items():
            if col_name == self.target:
                col_info["suggested_role"] = "target"

            if self.id_col and col_name == self.id_col:
                col_info["suggested_role"] = "id"

        if use_llm:
            p = enrich_with_llm(p, self.df)

        self.profile = p
        return p

    # =========================
    # GET FEATURES / TARGET
    # =========================
    def get_X_y(self):
        df = self.df.copy()

        # удаляем id
        if self.id_col and self.id_col in df.columns:
            df = df.drop(columns=[self.id_col])

        # удаляем datetime (если не используешь feature engineering)
        if self.datetime_col and self.datetime_col in df.columns:
            df = df.drop(columns=[self.datetime_col])

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X, y = self.clean_target(X, y)

        return X, y

    # =========================
    # SPLIT
    # =========================
    def split_data(self, test_size=0.2):
        from sklearn.model_selection import train_test_split

        X, y = self.get_X_y()

        # TIME SERIES SPLIT
        if self.datetime_col:
            df = self.df.sort_values(self.datetime_col).reset_index(drop=True)

            split_index = int(len(df) * (1 - test_size))

            train_df = df.iloc[:split_index]
            test_df = df.iloc[split_index:]

            X_train = train_df.drop(columns=[self.target])
            y_train = train_df[self.target]

            X_test = test_df.drop(columns=[self.target])
            y_test = test_df[self.target]

            return X_train, X_test, y_train, y_test

        # RANDOM SPLIT
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

    # =========================
    # ANALYZE
    # =========================
    def analyze(self):
        self.optimize_memory()

        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "missing": self.df.isnull().sum().to_dict(),
        }

        types = self.detect_types()
        target = self.detect_target()
        task = self.detect_task_type()
        id_col = self.detect_id_column()

        profile = self.run_profile(use_llm=False)

        info.update({
            "types": types,
            "target": target,
            "task": task,
            "datetime_col": self.datetime_col,
            "id_col": id_col,
            "profile": profile,
        })

        return info