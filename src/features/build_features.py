import joblib
import unittest
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

import logging

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocessing")


def filter_valid_dates(X):
    """
    Filters out rows with invalid or null dates in the 'track_album_release_date' column.

    Args:
        X (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing only valid dates.
    """
    X["track_album_release_date"] = pd.to_datetime(
        X["track_album_release_date"], errors="coerce"
    )
    valid_dates_mask = (
        X["track_album_release_date"].notna()
        & X["track_album_release_date"].dt.month.notnull()
        & X["track_album_release_date"].dt.day.notnull()
    )
    return X[valid_dates_mask]


def prepare_data(
    data,
    date_column="track_album_release_date",
    target="track_popularity",
    test_size=0.2,
):
    data = filter_valid_dates(data)  # Filter out invalid dates

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

    data.dropna(inplace=True)

    X = data.drop(columns=[target])
    y = pd.cut(
        data[target], bins=[-1, 20, 50, 80, 101], labels=[0, 1, 2, 3], right=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


### Feature engineering
class TopArtistTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_top_artists=10):
        self.num_top_artists = num_top_artists
        self.top_artists = None

    def fit(self, X, y=None):
        X["release_date__year"] = pd.to_datetime(
            X["track_album_release_date"], errors="coerce"
        ).dt.year
        songs_last_decade = X[
            X["release_date__year"] >= X["release_date__year"].max() - 10
        ]

        top_artists = (
            songs_last_decade.groupby("track_artist")
            .agg({"track_id": "count"})
            .rename(
                columns={
                    "track_id": "number_of_tracks",
                }
            )
        )

        top_artists = top_artists.sort_values(
            ["number_of_tracks"], ascending=[False]
        ).head(self.num_top_artists)

        self.top_artists = top_artists.index

        return self

    def transform(self, X):
        is_top_artist = X["track_artist"].isin(self.top_artists)
        return is_top_artist.to_frame(name="is_top_artist")

    def get_feature_names_out(self, input_features=None):
        return ["is_top_artist"]


class ReleaseDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.to_datetime(X, errors="coerce")

        month = X.dt.month
        day = X.dt.day

        month_season = month.map(
            {
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
                12: "Winter",
            }
        )

        day_category = pd.cut(
            day,
            bins=[0, 10, 20, 31],
            labels=["First 10", "Middle 10", "Last 10"],
            right=False,
        )

        return np.column_stack([month_season, day_category])

    def get_feature_names_out(self, input_features=None):
        return ["month_season", "day_category"]


class RemixOrCollabTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_remix = X.str.contains("remix", case=False).astype(int)
        is_collab = X.str.contains(r"(feat|ft\.|\(with)", case=False).astype(int)
        return np.column_stack([is_remix, is_collab])

    def get_feature_names_out(self, input_features=None):
        return ["is_remix", "is_collab"]


class WeekendTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.dropna()
        X = pd.to_datetime(X, errors="coerce")
        return X.dt.dayofweek.isin([5, 6]).astype(int).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["is_weekend"]


release_date_pipeline = make_pipeline(
    ReleaseDateTransformer(),
    OneHotEncoder(handle_unknown="ignore"),
)


num_pipeline = make_pipeline(
    StandardScaler(),
)

feature_engineering = ColumnTransformer(
    [
        (
            "release_date",
            release_date_pipeline,
            "track_album_release_date",
        ),
        (
            "release_day",
            WeekendTransformer(),
            "track_album_release_date",
        ),
        (
            "top_artist",
            TopArtistTransformer(num_top_artists=50),
            [
                "track_artist",
                "track_album_release_date",
                "track_id",
            ],
        ),
        (
            "genres",
            OneHotEncoder(handle_unknown="ignore"),
            ["playlist_genre", "playlist_subgenre"],
        ),
        (
            "track_name",
            RemixOrCollabTransformer(),
            "track_name",
        ),
        (
            "numerical",
            num_pipeline,
            [
                "danceability",
                "energy",
                "loudness",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "duration_ms",
            ],
        ),
        (
            "key",
            OrdinalEncoder(),
            ["key"],
        ),
        ("mode", "passthrough", ["mode"]),
        ("drop", "drop", "track_album_release_date"),
    ],
    remainder="drop",
)

final_pipeline = make_pipeline(feature_engineering)


## Causal Infernece Pipeline


class CausalInferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove duplicate tracks
        X = (
            X.sort_values("track_popularity", ascending=False)
            .drop_duplicates("track_name")
            .sort_index()
        )

        # Filter positive popularity
        X = X[X["track_popularity"] > 0]

        # Convert duration to mins
        X["duration_mins"] = X["duration_ms"] / 60000
        X["duration_mins"] = X["duration_mins"].round(2)
        X = X.drop(columns=["duration_ms"])

        # Extract release year
        X["track_album_release_date"] = pd.to_datetime(
            X["track_album_release_date"], errors="coerce"
        )
        X["release_year"] = X["track_album_release_date"].dt.year

        # Drop unwanted columns
        X = X.drop(
            columns=[
                "track_id",
                "track_name",
                "track_album_id",
                "track_album_name",
                "track_album_release_date",
                "playlist_name",
                "playlist_id",
            ]
        )
        return X

    def get_feature_names_out(self, input_features=None):
        return ["duration_mins", "release_year"]


# ### Unit Test
class TestTopArtistTransformer(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.data = pd.DataFrame(
            {
                "track_album_release_date": [
                    "2010-01-01",
                    "2011-05-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2019-01-01",
                ],
                "track_artist": ["Artist1", "Artist1", "Artist2", "Artist3", "Artist2"],
                "track_id": [1, 2, 3, 4, 5],
            }
        )
        self.transformer = TopArtistTransformer(num_top_artists=2)

    def test_fit(self):
        # Test the fitting process
        self.transformer.fit(self.data)
        self.assertEqual(len(self.transformer.top_artists), 2)
        self.assertIn("Artist2", self.transformer.top_artists)
        self.assertIn("Artist1", self.transformer.top_artists)

    def test_transform(self):
        # Fit and then transform the data
        self.transformer.fit(self.data)
        transformed = self.transformer.transform(self.data)
        expected = pd.DataFrame({"is_top_artist": [True, True, True, False, True]})
        pd.testing.assert_frame_equal(transformed, expected)

    def test_feature_names_out(self):
        # Check the output feature names
        output_names = self.transformer.get_feature_names_out()
        self.assertEqual(output_names, ["is_top_artist"])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTopArtistTransformer)
    unittest.TextTestRunner().run(suite)

    logger.info("Running train-test split...")

    # Load your data
    data = pd.read_csv("data/raw/spotify_songs_train.csv")

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = prepare_data(data)

    logger.info(f"Shape: {X_train.shape}")
    logger.info(f"Columns: {X_train.columns}")

    # Save the train and test data as a joblib file
    with open("data/interim/train_test_data.joblib", "wb") as f:
        joblib.dump((X_train, X_test, y_train, y_test), f)

    logger.info("Train-test executed completed successfully")

    final_pipeline.fit(X_train)

    with open("src/features/preprocessing.joblib", "wb") as f:
        joblib.dump(final_pipeline, f)

    logger.info("Preprocessing pipeline saved successfully")

    causal_inference_pipeline = make_pipeline(
        CausalInferenceTransformer(),
    )

    with open("src/features/causal_inference_pipeline.joblib", "wb") as file:
        joblib.dump(causal_inference_pipeline, file)
