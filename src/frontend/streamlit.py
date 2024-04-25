import streamlit as st
import pandas as pd
import joblib
import os
import sys
import torch
import torch.nn as nn

sys.path.append("src\\features")

from build_features import (
    TopArtistTransformer,
    ReleaseDateTransformer,
    RemixOrCollabTransformer,
    WeekendTransformer,
    CausalInferenceTransformer,
)

from classifier import ClassificationModel

# Load the preprocessing pipeline
preprocessing_pipeline = joblib.load("./src/features/preprocessing.joblib")
causal_inference_pipeline = joblib.load(
    "./src/features/causal_inference_pipeline.joblib"
)

# Load the trained models
xgb_model = joblib.load("./models/xgb_optuna_model.joblib")
dl_model = joblib.load("./models/deep_learning_optuna_model.joblib")


def predict_dl(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data)
        return preds.argmax(dim=1).detach().cpu().numpy()


def convert_to_label(predicition):
    if predicition == 0:
        return "Low popularity"
    elif predicition == 1:
        return "Moderate popularity"
    elif predicition == 2:
        return "High popularity"
    elif predicition == 3:
        return "Very high popularity"
    else:
        return "Unknown"


# Streamlit app
def main():
    st.title("Spotify Songs Prediction App")

    # Upload data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(data)

        # Preprocess data
        preprocessed_data = preprocessing_pipeline.transform(data)

        # Make predictions
        xgb_predictions = xgb_model.predict(preprocessed_data)
        dl_predictions = predict_dl(dl_model, torch.tensor(preprocessed_data).float())

        # Combine predictions with original data
        data["xgb_predictions"] = xgb_predictions
        data["dl_predictions"] = dl_predictions

        data["xgb_label"] = [convert_to_label(pred) for pred in xgb_predictions]
        data['dl_label'] = [convert_to_label(pred) for pred in dl_predictions]

        # data["causal_features"] = causal_preprocessed_data.to_dict("records")

        # Display predictions
        st.write("Predictions:")
        st.write(data)
        st.header("Top 5 high popularity songs")
        top_songs = data.sort_values(by="dl_predictions", ascending=False).head(5)
        st.write(top_songs)

if __name__ == "__main__":
    main()
