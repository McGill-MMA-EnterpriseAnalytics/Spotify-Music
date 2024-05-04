import os
import h2o
import joblib  # Import joblib for saving the model
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt


def train_h2o_model(X_train_transformed, X_test_transformed, y_train, y_test, target_column="target", max_models=10, seed=42):
    """
    This function trains an H2O AutoML model and evaluates its performance.
    """
    # Start H2O cluster
    h2o.init()

    # Convert data to H2O format, ensuring target columns are properly converted
    # Add target column name to the datasets before conversion
    X_train_transformed[target_column] = y_train.reset_index(drop=True)
    X_test_transformed[target_column] = y_test.reset_index(drop=True)

    train = h2o.H2OFrame(X_train_transformed)
    test = h2o.H2OFrame(X_test_transformed)

    # Ensure the target is treated as a categorical variable
    train[target_column] = train[target_column].asfactor()
    test[target_column] = test[target_column].asfactor()

    # Run AutoML
    aml = H2OAutoML(max_models=max_models, seed=seed)
    aml.train(
        x=train.columns[:-
                        1], y=target_column, training_frame=train, leaderboard_frame=test
    )

    # Get the leader model
    leader = aml.leader

    # Evaluate the leader model
    performance = leader.model_performance(test)
    print(performance)

    # Stop H2O cluster
    h2o.cluster().shutdown()
    return aml


def plot_model_performance(lb_df, sort_by="logloss", ascending=True, save_path=None):
    """
    This function plots the model performance based on the given DataFrame and saves the figure if a path is provided.
    """
    lb_df_sorted = lb_df.sort_values(by=sort_by, ascending=ascending)

    plt.figure(figsize=(10, 8))  # Adjusted figure size to prevent chopping
    plt.barh(lb_df_sorted["model_id"], lb_df_sorted[sort_by], color="skyblue")
    plt.xlabel(sort_by.capitalize())
    plt.ylabel("Model ID")
    plt.title(f"Model Performance by {sort_by.capitalize()}")
    plt.gca().invert_yaxis()  # Invert y-axis to have the best model on top
    plt.tight_layout()  # Adjust layout to make room for the elements

    if save_path:
        # Added bbox_inches='tight' to prevent chopping
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    # Load dataset
    X_train_transformed, X_test_transformed, y_train, y_test = joblib.load(
        "./data/processed/train_test_transformed_data.joblib"
    )

    aml = train_h2o_model(X_train_transformed, X_test_transformed, y_train, y_test)
    
    plot_model_performance(aml.leaderboard.as_data_frame(), save_path=r"reports\figures\h2o_base_model_results.png")