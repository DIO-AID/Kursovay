from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = {"MAE": mae, "R2": r2, "predictions": predictions}

    return results
