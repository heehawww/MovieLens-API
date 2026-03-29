import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as c:
        yield c

# Status
def test_status(test_client):
    res = test_client.get("/status")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# Search endpoint
def test_search_found(test_client):
    res = test_client.get("/movies/search", params={"query": "toy"})
    assert res.status_code == 200
    results = res.json()
    assert len(results) > 0
    assert results[0]["movie_id"] == 1
    assert results[0]["title"] == "Toy Story (1995)"
    assert results[0]["genres"] == "Animation|Children's|Comedy"

def test_search_not_found(test_client):
    res = test_client.get("/movies/search", params={"query": "xyzxyzxyz"})
    assert res.status_code == 404
    assert res.json() == {"detail": "No movies found for 'xyzxyzxyz'"}

def test_search_exceed_length(test_client):
    res = test_client.get("/movies/search", params={"query": "x" * 101})
    assert res.status_code == 422


# Similar movies endpoint
def test_similar_valid(test_client):
    res = test_client.get("/movies/1/similar")
    assert res.status_code == 200
    assert len(res.json()) == 5

def test_similar_invalid(test_client):
    res = test_client.get("/movies/267/similar") # moivie_id 267 was removed during preprocessing due to 'unknown' genre
    assert res.status_code == 404

def test_similar_out_of_range(test_client):
    res = test_client.get("/movies/99999/similar")
    assert res.status_code == 422

def test_similar_invalid_type(test_client):
    res = test_client.get("/movies/abc/similar")
    assert res.status_code == 422


# Recommendations endpoint
def test_recommendations_valid(test_client):
    res = test_client.post("/recommendations/from-liked-movies", json={"movie_ids": [1, 50]})
    assert res.status_code == 200
    assert len(res.json()) == 5

def test_recommendations_not_found(test_client):
    # 267 removed during preprocessing
    res = test_client.post("/recommendations/from-liked-movies", json={"movie_ids": [267, 268]})
    assert res.status_code == 404
    assert "267" in res.json()["detail"]

def test_recommendations_out_of_range(test_client):
    res = test_client.post("/recommendations/from-liked-movies", json={"movie_ids": [1, 99999]})
    assert res.status_code == 404

def test_recommendations_too_few(test_client):
    res = test_client.post("/recommendations/from-liked-movies", json={"movie_ids": [1]})
    assert res.status_code == 422 

def test_recommendations_too_many(test_client):
    res = test_client.post("/recommendations/from-liked-movies", json={"movie_ids": [1,2,3,4,5,6]})
    assert res.status_code == 422