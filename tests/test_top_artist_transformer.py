import unittest
import pandas as pd
# import sys
# sys.path.append('../src/features')
from src.features.build_features import (
    TopArtistTransformer,
    # ReleaseDateTransformer,
    # RemixOrCollabTransforme,
    # WeekendTransformer,
    # CausalInferenceTransformer,
)


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
        expected = pd.DataFrame(
            {"is_top_artist": [True, True, True, False, True]})
        pd.testing.assert_frame_equal(transformed, expected)

    def test_feature_names_out(self):
        # Check the output feature names
        output_names = self.transformer.get_feature_names_out()
        self.assertEqual(output_names, ["is_top_artist"])

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTopArtistTransformer)
    unittest.TextTestRunner().run(suite)