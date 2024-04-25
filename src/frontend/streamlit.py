import streamlit as st
import requests

import pandas as pd

BASE_URL = "https://insy695-zlcw5o425a-nn.a.run.app"


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

        # Convert data to JSON for prediction
        json_data = data.to_json(orient="records")

        # Define the URLs of the FastAPI endpoints
        xgb_prediction_url = f"{BASE_URL}/xgb"
        dl_prediction_url = f"{BASE_URL}/dl"

        # Send the data to the XGB FastAPI endpoint for prediction
        headers = {"Content-Type": "application/json"}
        xgb_response = requests.post(
            xgb_prediction_url, data=json_data, headers=headers
        )

        # Send the data to the DL FastAPI endpoint for prediction
        dl_response = requests.post(dl_prediction_url, data=json_data, headers=headers)

        if xgb_response.status_code == 200 and dl_response.status_code == 200:
            xgb_predictions = xgb_response.json()["predictions"]
            dl_predictions = dl_response.json()["predictions"]

            # Combine predictions with original data
            data["xgb_predictions"] = xgb_predictions
            data["dl_predictions"] = dl_predictions

            data["xgb_label"] = [convert_to_label(pred) for pred in xgb_predictions]
            data["dl_label"] = [convert_to_label(pred) for pred in dl_predictions]

            # Display predictions
            st.write("Predictions:")
            st.write(data)
            st.header("Top 5 high popularity songs")
            top_songs = data.sort_values(by="dl_predictions", ascending=False).head(5)
            st.write(top_songs)
        else:
            st.error("Failed to get predictions from the models.")


if __name__ == "__main__":
    main()
