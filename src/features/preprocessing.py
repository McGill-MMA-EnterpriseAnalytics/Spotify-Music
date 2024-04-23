#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import time
import warnings
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
#from lightgbm import LGBMClassifier
from rich import print
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    learning_curve
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

get_ipython().run_line_magic('load_ext', 'rich')
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")


# In[25]:


path ="../../data/raw/spotify_songs_train.csv"




# In[26]:


def prepare_data(data, date_column='track_album_release_date', target='track_popularity', test_size=0.2):
  
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    not_null_mask = data[date_column].dt.month.notnull() & data[date_column].dt.day.notnull()
    data = data[not_null_mask]
    data = data.dropna()
    X = data.drop(columns=[target])
    y = (data[target] > 50).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


data = pd.read_csv(path)  
X_train, X_test, y_train, y_test = prepare_data(data)


# In[27]:


print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# ## Feature engineering
# 

# In[28]:


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


# In[29]:


def get_num_playlists(X):
    num_playlist = X.groupby("track_id")["playlist_id"].transform("nunique").values

    return np.column_stack([num_playlist])


def playlist_name(X, feature_names):
    return ["num_playlists"]


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

    day_category = pd.cut(day, bins=[0, 10, 20, 31], labels=["First 10", "Middle 10", "Last 10"], right=False)

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
    X = pd.to_datetime(X, errors="coerce")
    return X.dt.dayofweek.isin([5, 6]).astype(int).values.reshape(-1, 1)


def is_weekend_name(X, feature_names):
    return ["is_weekend"]


num_playlist_pipeline = make_pipeline(
    FunctionTransformer(
        get_num_playlists,
        validate=False,
        feature_names_out=playlist_name,
    ),
    StandardScaler(),
)

release_date_pipeline = make_pipeline(
    FunctionTransformer(
        release_date, validate=False, feature_names_out=release_date_name
    ),
    OneHotEncoder(handle_unknown="ignore"),
)

num_pipeline = make_pipeline(
    StandardScaler(),
)


feature_engineering = ColumnTransformer(
    [
        # (
        #     "num_playlists",
        #     num_playlist_pipeline,
        #     ["track_id", "playlist_id"],
        # ),
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


# In[30]:


feature_engineering.fit(X_train)
with open('feature_engineering.pkl', 'wb') as f:
    pickle.dump(feature_engineering, f)

with open('train_test_data.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)


# ### Unit Test

# In[32]:


import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
class TestTopArtistTransformer(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.data = pd.DataFrame({
            'track_album_release_date': ['2010-01-01', '2011-05-01', '2020-01-01', '2020-01-02', '2019-01-01'],
            'track_artist': ['Artist1', 'Artist1', 'Artist2', 'Artist3', 'Artist2'],
            'track_id': [1, 2, 3, 4, 5]
        })
        self.transformer = TopArtistTransformer(num_top_artists=2)

    def test_fit(self):
    # Test the fitting process
        self.transformer.fit(self.data)
        self.assertEqual(len(self.transformer.top_artists), 2)
        self.assertIn('Artist2', self.transformer.top_artists)
        self.assertIn('Artist1', self.transformer.top_artists)

    def test_transform(self):
        # Fit and then transform the data
        self.transformer.fit(self.data)
        transformed = self.transformer.transform(self.data)
        expected = pd.DataFrame({
            'is_top_artist': [True, True, True, False, True]
        })
        pd.testing.assert_frame_equal(transformed, expected)

    def test_feature_names_out(self):
        # Check the output feature names
        output_names = self.transformer.get_feature_names_out()
        self.assertEqual(output_names, ['is_top_artist'])


suite = unittest.TestLoader().loadTestsFromTestCase(TestTopArtistTransformer)
unittest.TextTestRunner().run(suite)



# In[ ]:



