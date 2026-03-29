# Movie Recommendation API

A movie recommendation system built with FastAPI and Streamlit, using SVD-based
collaborative filtering and IDF-weighted genre scoring on the MovieLens 100K dataset.

---

## Methodology

### Data Preprocessing
- Movies with `unknown` genre tags are removed
- Ratings are filtered to only include valid movie IDs
- User-item matrix is mean-centered per user to remove rating bias

### Recommendation Model
- **SVD** (k=20 latent factors) is applied to the mean-centered user-item matrix to
  address 93.7% sparsity, capturing latent user-movie preference patterns
- **Cosine similarity** is computed on movie latent vectors for similar movie lookup
- **IDF-weighted genre scoring** is combined with SVD cosine similarity scores in a
  70/30 split for the blended recommendations endpoint — downweighting common genres
  like Drama in favour of more distinctive genres like Sci-Fi

---

## API Endpoints
 **Rate limiting** is enforced per IP address. Search and similar endpoints allow 30 requests/minute. 
  Recommendations allow 20 requests/minute. Exceeding the limit returns a `429 Too Many Requests` response.

### `GET /status`
Health check.

**Response**
```json
{"status": "ok"}
```

---

### `GET /movies/search?query=`
Search for movies by title (case-insensitive substring match).

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | Yes | Title or partial title to search for (max 100 characters) |

**Example**
```
GET /movies/search?query=toy
```

**Response**
```json
[
  {
    "movie_id": 1,
    "title": "Toy Story (1995)",
    "genres": "Animation|Children's|Comedy"
  }
]
```

**Errors**
| Status | Description |
|--------|-------------|
| 404 | No movies found matching the query |
| 422 | Query is empty or exceeds 100 characters |
| 429 | Rate limit exceeded (30 requests/minute) |

---

### `GET /movies/{movie_id}/similar`
Returns the top 5 most similar movies based on SVD cosine similarity.

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| movie_id | integer | Yes | Movie ID (1–1682) |

**Example**
```
GET /movies/1/similar
```

**Response** - returns a list of 5 movies ordered by final score descending (only one shown below).
```json
[
  {
    "movie_id": 364,
    "title": "Lion King, The (1994)",
    "similarity_score": 0.9231,
    "genres": "Animation|Children's|Musical"
  }
]
```

**Errors**
| Status | Description |
|--------|-------------|
| 404 | Movie ID not found |
| 422 | Movie ID out of range or invalid type |
| 429 | Rate limit exceeded (30 requests/minute) |

---

### `POST /recommendations/from-liked-movies`
Returns top 5 personalised recommendations based on a blend of SVD similarity
and IDF-weighted genre profile matching.

**Request Body**
```json
{
  "movie_ids": [1, 50, 172]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| movie_ids | list[int] | Yes | 2–5 movie IDs |

**Response** - returns a list of 5 movies ordered by final score descending (only one shown below).
```json
[
  {
    "movie_id": 423,
    "title": "E.T. the Extra-Terrestrial (1982)",
    "final_score": 0.6842,
    "genres": "Children's|Drama|Fantasy|Sci-Fi"
  }
]
```

**Errors**
| Status | Description |
|--------|-------------|
| 404 | One or more movie IDs not found |
| 422 | Less than 2 or more than 5 movie IDs provided |
| 429 | Rate limit exceeded (20 requests/minute) |

---

## Setup

### Option 1 — Docker 
Requires [Docker Desktop](https://www.docker.com/products/docker-desktop).
```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

### Option 2 — Local
Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).
```bash
uv sync
uv run python run.py
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

---

## Tests
```bash
uv run pytest tests/ -v
```

---

## Project Structure
```
movie_api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── schemas.py         # Pydantic request/response schemas
│   └── frontend.py        # Streamlit frontend
├── data/
│   ├── raw/               # original MovieLens 100K files
│   └── movie_genre_vectors.csv     # preprocessed datasets
│   └── movies_metadata.csv
│   └── similarity_matrix.csv
│   └── top_similar_movies.csv     
├── preprocess/
│   └── preprocessing.py   # data preprocessing pipeline
│   └── eda.ipynb          # exploratory data analysis of original dataset 
├── tests/
│   ├── __init__.py
│   ├── integration_test.py
│   └── unit_test.py
├── Dockerfile
├── docker-compose.yml
├── run.py                 # local development runner
└── pyproject.toml
└── uv.lock
```
