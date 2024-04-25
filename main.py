import json

import joblib
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from torch.nn import functional as F

from src.features.build_features import (
    CausalInferenceTransformer,
    ReleaseDateTransformer,
    RemixOrCollabTransformer,
    TopArtistTransformer,
    WeekendTransformer,
)
from src.models.train_lightning import ClassificationModel


class SampleRequest(BaseModel):
    track_id: str
    track_name: str
    track_artist: str
    track_popularity: int
    track_album_id: str
    track_album_name: str
    track_album_release_date: str
    playlist_name: str
    playlist_id: str
    playlist_genre: str
    playlist_subgenre: str
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int


app = FastAPI()

preprocessing = joblib.load("./src/features/preprocessing.joblib")
xgb_model = joblib.load("./models/xgb_optuna_model.joblib")
dl_model = joblib.load("./models/deep_learning_optuna_model.joblib")


@app.post("/xgb")
async def xgb_predict(request: list[SampleRequest]):
    df = pd.DataFrame([i.model_dump() for i in request])
    df = preprocessing.transform(df)

    preds = xgb_model.predict(df)

    return {"predictions": preds.tolist()}


@app.post("/dl")
async def dl_predict(request: list[SampleRequest]):
    df = pd.DataFrame([i.model_dump() for i in request])
    df = preprocessing.transform(df)
    input_tensor = torch.tensor(df, dtype=torch.float32)

    preds = dl_model(input_tensor)

    preds = F.softmax(preds, dim=1).argmax(dim=1).detach().numpy()

    return {"predictions": preds.tolist()}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Spotify Song Popularity API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
