import numpy as np
import pytest
import sys
import os
import pandas as pd
from app.main import get_genre_profile, data


@pytest.fixture(autouse=True)
def load_data():
    data["genre_vectors"] = pd.read_csv("data/movie_genre_vectors.csv", index_col="movie_id")
    data["genre_vectors"].index = data["genre_vectors"].index.astype(int)

def test_genre_profile_sums_to_one():
    profile = get_genre_profile([1, 50])
    assert abs(profile.sum() - 1.0) < 1e-6

def test_genre_profile_is_ndarray():
    profile = get_genre_profile([1])
    assert isinstance(profile, np.ndarray)

def test_genre_profile_invalid_ids():
    profile = get_genre_profile([99999])
    assert profile.sum() == 0.0