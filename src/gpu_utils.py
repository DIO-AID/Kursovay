# gpu_utils.py
"""
Утилиты для GPU/CPU обучения моделей.
Просто импортируй этот файл и вызывай функцию `get_gpu_params()`.
"""

import os

# ===============================
# Настройки системы
# ===============================
DEFAULT_CPU_JOBS = 4  # сколько потоков CPU использовать
DEFAULT_GPU_ID = 0  # какой GPU использовать, если доступен


def get_gpu_params(
    model_name="xgb",
    tree_method="hist",  # изменено с gpu_hist на hist для легкого режима по умолчанию
    n_jobs=DEFAULT_CPU_JOBS,
    gpu_id=DEFAULT_GPU_ID,
):
    """
    Возвращает словарь параметров для GPU/CPU обучения для разных моделей.

    model_name: "xgb", "lgbm", "catboost"
    """
    params = {}

    # Проверяем доступность GPU через переменную среды (можно улучшить через pynvml)
    use_gpu = os.environ.get("USE_GPU", "0") == "1"  # изменено с "1" на "0" - по умолчанию CPU

    if model_name.lower() == "xgb":
        if use_gpu:
            params = {
                "tree_method": "gpu_hist",  # GPU режим
                "gpu_id": gpu_id,
                "n_jobs": n_jobs,
                "verbosity": 1,
            }
        else:
            params = {"tree_method": "hist", "n_jobs": n_jobs, "verbosity": 1}
    elif model_name.lower() == "lgbm":
        if use_gpu:
            params = {
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": gpu_id,
                "n_jobs": n_jobs,
                "verbose": -1,
            }
        else:
            params = {"device": "cpu", "n_jobs": n_jobs, "verbose": -1}
    elif model_name.lower() == "catboost":
        if use_gpu:
            params = {
                "task_type": "GPU",
                "devices": str(gpu_id),
                "thread_count": n_jobs,
                "verbose": 100,
            }
        else:
            params = {"task_type": "CPU", "thread_count": n_jobs, "verbose": 100}
    else:
        # по умолчанию CPU
        params = {"n_jobs": n_jobs}

    return params


def set_env_gpu(gpu_id=DEFAULT_GPU_ID):
    """
    Настройка переменных окружения для GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)