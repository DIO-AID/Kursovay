from pydantic import BaseModel
from typing import Optional, List


class ExperimentConfig(BaseModel):
    model_name: str
    mode: str = "grid"
    target: str = "median_house_value"
    use_feature: bool = False
    use_shap: bool = False
    n_trials: int = 100
    cv_folds: int = 5
    random_state: int = 42
    timeout: int = 600
    study_name: Optional[str] = None


class ExperimentResult(BaseModel):
    model_name: str
    method: str
    val_r2_mean: float
    val_r2_std: float
    test_r2: float
    train_r2: float
    n_params: Optional[int] = None
    time_seconds: float
    best_params: dict = {}
