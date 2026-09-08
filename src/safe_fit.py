def safe_fit(model, X, y, fit_params=None):
    if not fit_params:
        return model.fit(X, y)

    params = dict(fit_params)
    model_name = model.__class__.__name__.lower()

    # =====================================================
    # XGBOOST (FIXED STRICT MODE)
    # =====================================================
    if "xgb" in model_name or "xgboost" in model_name:

        params.pop("verbose", None)

        es = params.pop("early_stopping_rounds", None)

        if es is not None:
            try:
                from xgboost import callback

                params["callbacks"] = [
                    callback.EarlyStopping(rounds=es, save_best=True)
                ]
            except Exception:
                pass

        # 🔥 CRITICAL: XGB sklearn API may still reject extra params
        allowed = model.fit.__code__.co_varnames
        params = {k: v for k, v in params.items() if k in allowed}

        return model.fit(X, y, **params)

    # =====================================================
    # LIGHTGBM
    # =====================================================
    if "lgbm" in model_name or "lightgbm" in model_name:

        params.pop("verbose", None)

        es = params.pop("early_stopping_rounds", None)

        if es is not None:
            try:
                import lightgbm as lgb
                callbacks = params.get("callbacks", [])
                callbacks.append(lgb.early_stopping(es, verbose=False))
                params["callbacks"] = callbacks
            except Exception:
                pass

        return model.fit(X, y, **params)

    # =====================================================
    # CATBOOST
    # =====================================================
    if "catboost" in model_name:

        safe = {}
        allowed = model.fit.__code__.co_varnames

        for k, v in params.items():
            if k in allowed:
                safe[k] = v

        return model.fit(X, y, **safe)

    # =====================================================
    # FALLBACK
    # =====================================================
    return model.fit(X, y)