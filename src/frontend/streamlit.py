import os

import pandas as pd
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from supabase import Client, create_client

load_dotenv(find_dotenv())

# Getting Supabase URL and Key from environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase_client = create_client(supabase_url, supabase_key)

if "top_song_list" not in st.session_state:
    st.session_state.top_song_list = []


def convert_to_label(prediction):
    labels = {
        0: "Low popularity",
        1: "Moderate popularity",
        2: "High popularity",
        3: "Very high popularity",
    }
    return labels.get(prediction, "Unknown")


BASE_URL = "https://insy695-zlcw5o425a-nn.a.run.app"


# Streamlit app
def main():
    st.title("Spotify Songs Prediction App")

    # Upload data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(data.head())

        # Convert data to JSON for prediction
        json_data = data.to_json(orient="records")

        st.subheader("Model Prediction")
        model_call_form = st.form(key="model_call_form")

        # Number of songs to recommend
        num_songs = model_call_form.slider(
            "Number of songs to recommend", min_value=1, max_value=10, value=5
        )
        # Allow user to select the model
        model_choice = model_call_form.selectbox(
            "Select the model", ["XGBoost", "Deep Learning"]
        )

        run_model = model_call_form.form_submit_button("Run Model")

        if run_model:
            if model_choice == "XGBoost":
                # Use the XGBoost model for prediction
                xgb_prediction_url = f"{BASE_URL}/xgb"
                headers = {"Content-Type": "application/json"}
                xgb_response = requests.post(
                    xgb_prediction_url, data=json_data, headers=headers
                )

                if xgb_response.status_code == 200:
                    xgb_predictions = xgb_response.json()["predictions"]
                    data["Prediction"] = xgb_predictions
                    data["Label"] = [convert_to_label(pred) for pred in xgb_predictions]
                    data["Model Used"] = "XGBoost"
            else:
                # Use the Deep Learning model for prediction
                dl_prediction_url = f"{BASE_URL}/dl"
                headers = {"Content-Type": "application/json"}
                dl_response = requests.post(
                    dl_prediction_url, data=json_data, headers=headers
                )

                if dl_response.status_code == 200:
                    dl_predictions = dl_response.json()["predictions"]
                    data["Prediction"] = dl_predictions
                    data["Label"] = [convert_to_label(pred) for pred in dl_predictions]
                    data["Model Used"] = "Deep Learning"

            # Display the predictions

            data_preds = data.copy()
            st.session_state["data_preds"] = data_preds

            top_songs = (
                data[
                    [
                        "track_name",
                        "track_artist",
                        "Prediction",
                        "Label",
                        "Model Used",
                    ]
                ]
                .sort_values(by="Prediction", ascending=False)
                .head(num_songs)
            )

            st.write("Top Songs:")
            st.write(top_songs)

            # Save the top songs to session state
            st.session_state.top_song_list = top_songs["track_name"].tolist()

        # Ask user for feedback
        if st.session_state.top_song_list:
            st.subheader("Feedback")
            with st.form("feedback_form"):
                st.write("Please provide feedback on the predictions:")
                feedback_song = st.selectbox(
                    "Select a song", st.session_state.top_song_list
                )

                feedback_category = st.selectbox(
                    "Select the correct category",
                    options=[0, 1, 2, 3],
                    format_func=convert_to_label,
                )

                feedback_submit = st.form_submit_button("Submit Feedback")

                if feedback_submit:
                    track_data = (
                        st.session_state.data_preds[
                            st.session_state.data_preds["track_name"] == feedback_song
                        ]
                        .iloc[0]
                        .to_dict()
                    )

                    track_data.pop("Prediction")
                    track_data.pop("Label")
                    track_data.pop("Model Used")
                    track_data["feedback_track_popularity"] = feedback_category

                    # Save to Supabase

                    insert_response = (
                        supabase_client.table("feedback").insert([track_data]).execute()
                    )
                    if insert_response is not None:
                        st.success("Feedback submitted successfully!")
                    else:
                        st.error("Failed to submit feedback.")


if __name__ == "__main__":
    main()
