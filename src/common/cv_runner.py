import numpy as np
import time
import logging
from typing import Callable, Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

logger = logging.getLogger(__name__)


def cv_evaluate(
    model,
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
    scoring: str = "r2",
    n_jobs: int = 1,
) -> dict:
    """
    Единая CV-оценка модели. Возвращает dict с mean/std по каждому сплиту.
    """
    strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    r2_scores = cross_val_score(
        model, X, y, cv=strategy, scoring="r2", n_jobs=n_jobs
    )
    mae_scores = -cross_val_score(
        model, X, y, cv=strategy, scoring="neg_mean_absolute_error", n_jobs=n_jobs
    )

    return {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "mae_mean": float(mae_scores.mean()),
        "mae_std": float(mae_scores.std()),
        "n_folds": cv,
    }


def train_evaluate_split(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    refit: bool = True,
) -> dict:
    """
    Обучает модель на train, оценивает на test. Один раз — без подбора.
    refit=True: финальное обучение на train.
    """
    start = time.time()
    if refit:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    return {
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_time": float(train_time),
    }


def fit_preprocessor(preprocessor, X_train):
    """
    Фитит препроцессор ТОЛЬКО на train, возвращает transforms.
    """
    preprocessor.fit(X_train)
    return preprocessor


def full_pipeline_evaluate(
    model_factory: Callable,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor=None,
    params: Optional[dict] = None,
) -> dict:
    """
    Полный цикл: создание модели → (опционально) fit препроцессора → train/test оценка.
    """
    if params:
        model = model_factory(**params)
    else:
        model = model_factory()

    if preprocessor is not None:
        from sklearn.pipeline import Pipeline
        full_model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
    else:
        full_model = model

    results = train_evaluate_split(full_model, X_train, X_test, y_train, y_test)

    return results
