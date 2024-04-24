import pickle
import unittest
import warnings

import numpy as np
import pandas as pd

# from lightgbm import LGBMClassifier
from rich import print
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")


def filter_valid_dates(X):
    """
    Filters out rows with invalid or null dates in the 'track_album_release_date' column.
    
    Args:
        X (pandas.DataFrame): The input DataFrame.
        
    Returns:
        pandas.DataFrame: The filtered DataFrame containing only valid dates.
    """
    X['track_album_release_date'] = pd.to_datetime(X['track_album_release_date'], errors='coerce')
    valid_dates_mask = X['track_album_release_date'].notna() & \
                       X['track_album_release_date'].dt.month.notnull() & \
                       X['track_album_release_date'].dt.day.notnull()
    return X[valid_dates_mask]


def prepare_data(data, date_column='track_album_release_date', target='track_popularity', test_size=0.2):
    data = filter_valid_dates(data)  # Filter out invalid dates
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data.dropna(inplace=True)
    X = data.drop(columns=[target])
    y = pd.cut(data[target], bins=[-1, 20, 50, 80, 101], labels=[1, 2, 3, 4], right=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Load your data
data = pd.read_csv("data/raw/spotify_songs_train.csv")
# Split the data into train and test sets
X_train, X_test, y_train, y_test = prepare_data(data)


print(X_train.shape)
print(X_train.columns)
# Save the train and test data as a pickle file
with open("train_test_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
# ## Feature engineering
#
def convert_and_filter_dates(X):
    X = pd.to_datetime(X, errors="coerce")
    return X

date_transformer = FunctionTransformer(convert_and_filter_dates, validate=False)


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


# def get_num_playlists(X):
#     num_playlist = X.groupby("track_id")["playlist_id"].transform("nunique").values

#     return np.column_stack([num_playlist])


# def playlist_name(X, feature_names):
#     return ["num_playlists"]

# def convert_to_datetime(data, date_column):
#     print("hello1")
#     print(data[date_column])
#     data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
#     print("hello2")
    
#     return data

# def filter_null_dates(data, date_column):
    
#     print("hello3")
#     not_null_mask = (data[date_column].dt.month.notnull() & data[date_column].dt.day.notnull())
#     print("hello4")
    
#     data = data[not_null_mask]
#     print("hello5")
    
#     return data

def convert_and_filter_dates(X):
    X = pd.to_datetime(X, errors="coerce")
    # Filter out NaT values along with ensuring month and day are not null
    valid_dates_mask = X.notna() & X.dt.month.notnull() & X.dt.day.notnull()
    filtered_dates = X[valid_dates_mask]
    print(filtered_dates)
    return filtered_dates


def release_date(X):
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


def release_date_name(X, feature_names):
    return ["month_season", "day_category"]


def get_is_remix_or_collab(X):
    is_remix = X.str.contains("remix", case=False).astype(int)
    is_collab = X.str.contains(r"(feat|ft\.|\(with)", case=False).astype(int)
    return np.column_stack([is_remix, is_collab])


def is_remix_or_collab_name(X, feature_names):
    return ["is_remix", "is_collab"]


def get_is_weekend(X):
    X = X.dropna()
    X = pd.to_datetime(X, errors="coerce")
    return X.dt.dayofweek.isin([5, 6]).astype(int).values.reshape(-1, 1)


def is_weekend_name(X, feature_names):
    return ["is_weekend"]

def drop_release_date_column(X):
    return X.drop(columns=['track_album_release_date'])



# num_playlist_pipeline = make_pipeline(
#     FunctionTransformer(
#         get_num_playlists,
#         validate=False,
#         feature_names_out=playlist_name,
#     ),
#     StandardScaler(),
# )

release_date_pipeline = make_pipeline(
    FunctionTransformer(
        release_date, validate=False, feature_names_out=release_date_name
    ),
    OneHotEncoder(handle_unknown="ignore"),
)




num_pipeline = make_pipeline(
    StandardScaler(),
)

# date_transformer = make_pipeline(
#     FunctionTransformer(lambda X: filter_null_dates(convert_to_datetime(X, date_column="track_album_release_date")), validate=False)
# )

date_transformer = FunctionTransformer(
    lambda X: convert_and_filter_dates(X).values.reshape(-1, 1),
    validate=False
    #kw_args={"date_column": "track_album_release_date"}
)


# feature_engineering = ColumnTransformer(
#     [
#         ("date", date_transformer, "track_album_release_date"),
#         ("release_date", release_date_pipeline, "track_album_release_date"),
#         ("release_day", FunctionTransformer(get_is_weekend, validate=False, feature_names_out=is_weekend_name), "track_album_release_date"),
#         ("top_artist", TopArtistTransformer(num_top_artists=50), ["track_artist", "track_album_release_date", "track_id"]),
      



feature_engineering = ColumnTransformer(
        [
            # (
            #     "num_playlists",
            #     num_playlist_pipeline,
            #     ["track_id", "playlist_id"],
            # ),
            ("date",
            date_transformer,
            "track_album_release_date"
            ),

        
        (
            "release_date",
            release_date_pipeline,
            "track_album_release_date",
        ),
        (
            "release_day",
            FunctionTransformer(
                get_is_weekend, validate=False, feature_names_out=is_weekend_name
            ),
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
            FunctionTransformer(
                get_is_remix_or_collab,
                validate=False,
                feature_names_out=is_remix_or_collab_name,
            ),
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
    ],
    remainder="drop",
)

print(X_train.columns)

final_pipeline = make_pipeline(
    feature_engineering,
    FunctionTransformer(drop_release_date_column, validate=False)
)


final_pipeline.fit(X_train)
with open("./src/features/preprocessing.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)



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


suite = unittest.TestLoader().loadTestsFromTestCase(TestTopArtistTransformer)
unittest.TextTestRunner().run(suite)









## Causal Infernece Pipeline


def remove_duplicate_tracks(df):
    return (
        df.sort_values("track_popularity", ascending=False)
        .drop_duplicates("track_name")
        .sort_index()
    )


def filter_positive_popularity(df):
    return df[df["track_popularity"] > 0]


def convert_duration_to_mins(df):
    df["duration_mins"] = df["duration_ms"] / 60000
    df["duration_mins"] = df["duration_mins"].round(2)
    return df.drop(columns=["duration_ms"])


def extract_release_year(df):
    df["track_album_release_date"] = pd.to_datetime(
        df["track_album_release_date"], errors="coerce"
    )
    df["release_year"] = df["track_album_release_date"].dt.year
    return df


def drop_unwanted_columns(df):
    return df.drop(
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


causal_inference_pipeline = make_pipeline(
    FunctionTransformer(remove_duplicate_tracks, validate=False),
    FunctionTransformer(filter_positive_popularity, validate=False),
    FunctionTransformer(convert_duration_to_mins, validate=False),
    FunctionTransformer(extract_release_year, validate=False),
    FunctionTransformer(drop_unwanted_columns, validate=False),
)

with open("./src/features/causal_inference_pipeline.pkl", "wb") as file:
    pickle.dump(causal_inference_pipeline, file)
