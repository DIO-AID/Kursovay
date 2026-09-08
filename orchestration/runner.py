from experiments.run_experiment import run_experiment
from configs.config import BASE_CONFIG, MODEL_CONFIGS


def run_all_experiments(df):

    experiments = {
        "BASE": {"feature": False, "shap": False, "mode": "base"},
        "FULL": {"feature": True, "shap": True, "mode": "optuna"},
        "STACK": {"feature": True, "shap": True, "mode": "stack"},
    }

    results = {}

    for name, cfg in experiments.items():

        print(f"\nRUNNING {name}")

        try:
            config = BASE_CONFIG.copy()
            config.update(cfg)

            if name != "STACK":
                model_class, space = MODEL_CONFIGS["xgb"]
                config.update(
                    {
                        "model_name": "xgb",
                        "model_class": model_class,
                        "search_space": space,
                        "study_name": f"{name}_study",
                    }
                )
            else:
                config.update(
                    {
                        "model_name": "xgb",
                        "study_name": f"{name}_study",
                    }
                )

            score = run_experiment(df, config)
            results[name] = score

        except Exception as e:
            print("ERROR:", e)
            results[name] = None

    return results