import pandas as pd
import joblib
from build_features import (
    TopArtistTransformer,
    ReleaseDateTransformer,
    RemixOrCollabTransformer,
    WeekendTransformer,
    CausalInferenceTransformer,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_pipeline")
pd.set_option("display.max_columns", None)


if __name__ == "__main__":
    data = pd.read_csv("data/raw/spotify_songs_train.csv")

    logger.info("Loading pipelines...")
    train_test_data = joblib.load("./data/interim/train_test_data.joblib")
    preprocessing = joblib.load("./src/features/preprocessing.joblib")
    causal_inference = joblib.load("./src/features/causal_inference_pipeline.joblib")

    X_train, X_test, y_train, y_test = train_test_data

    X_train_transformed = pd.DataFrame(
        preprocessing.fit_transform(X_train),
        columns=preprocessing.get_feature_names_out(),
    )
    X_test_transformed = pd.DataFrame(
        preprocessing.transform(X_test), columns=preprocessing.get_feature_names_out()
    )

    logger.info("Pickling transformed data...")
    with open("data/processed/train_test_transformed_data.joblib", "wb") as f:
        joblib.dump((X_train_transformed, X_test_transformed, y_train, y_test), f)
