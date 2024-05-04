import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.xgboost
import optuna
import joblib  # Import joblib for saving the model


# training xgboost model and perform hyperparameter tuning using optuna



def objective(trial):
    with mlflow.start_run(nested=True):
        param = {
            "verbosity": 0,
            "objective": "multi:softmax",
            "num_class": len(np.unique(y_train)),
            "eval_metric": "mlogloss",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float(
                "rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float(
                "skip_drop", 1e-8, 1.0, log=True)

        mlflow.log_params(param)

        dtrain = xgb.DMatrix(X_train_transformed, label=y_train)
        dvalid = xgb.DMatrix(X_test_transformed, label=y_test)

        bst = xgb.train(
            param, dtrain, evals=[(dvalid, "validation")], early_stopping_rounds=10
        )
        mlflow.xgboost.log_model(bst, "model")

        preds = bst.predict(dvalid, output_margin=True)
        pred_labels = np.argmax(preds, axis=1)
        accuracy = accuracy_score(y_test, pred_labels)
        mlflow.log_metric("accuracy", accuracy)
        return accuracy


if __name__ == "__main__":
    # Load dataset
    X_train_transformed, X_test_transformed, y_train, y_test = joblib.load(
    "./data/processed/train_test_transformed_data.joblib")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_trial.params
    print("Best trial:", study.best_trial.params)

    # Train final model with adjusted labels
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_transformed, y_train)

    # Save the model
    joblib.dump(final_model, "./models/xgb_optuna_model.joblib")
