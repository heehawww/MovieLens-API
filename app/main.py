from fastapi import FastAPI, HTTPException, Query, Path, Request
from contextlib import asynccontextmanager
from typing import Annotated
import pandas as pd
import numpy as np

from .schemas import LikedMovies, MovieResult, SimilarMovie, Recommendation

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


data = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    data["movies"]        = pd.read_csv("data/movies_metadata.csv", index_col="movie_id")
    data["top_similar"]   = pd.read_csv("data/top_similar_movies.csv")
    data["similarity_df"] = pd.read_csv("data/similarity_matrix.csv", index_col="movie_id")
    data["genre_vectors"] = pd.read_csv("data/movie_genre_vectors.csv", index_col="movie_id")
    data["similarity_df"].columns = data["similarity_df"].columns.astype(int)
    data["similarity_df"].index   = data["similarity_df"].index.astype(int)
    data["genre_vectors"].index   = data["genre_vectors"].index.astype(int)
    data["movies"].index          = data["movies"].index.astype(int)
    yield

app = FastAPI(title="Movie Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # only allow streamlit frontend
    allow_methods=["GET", "POST"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ── Helper ─────────────────────────────────────
# Create a user profile by averaging the genre vectors of the liked movies
def get_genre_profile(liked_ids: list[int]) -> np.ndarray:
    gv = data["genre_vectors"]
    valid = [m for m in liked_ids if m in gv.index]

    # If none of the liked movies have genre vectors, return a zero vector first 
    if not valid:
        return np.zeros(len(gv.columns))
    
    profile = sum(gv.loc[m].values for m in valid)
    total = profile.sum()
    return profile / total if total > 0 else profile

# ── Routes ─────────────────────────────────────
# Basic health check
@app.get("/status")
def status():
    return {"status": "ok"}


# Search movies by title (case-insensitive substring match)
@app.get("/movies/search", response_model=list[MovieResult])
@limiter.limit("30/minute")
def search(request: Request, query: Annotated[str, Query(min_length=1, max_length=100)]):
    movies = data["movies"]
    results = movies[movies["title"].str.contains(query, case=False, na=False)]
    if results.empty:
        raise HTTPException(404, f"No movies found for '{query}'")
    return results.reset_index()[["movie_id", "title", "genres"]].to_dict(orient="records")


# Get top 5 similar movies based on SVD similarity
@app.get("/movies/{movie_id}/similar", response_model=list[SimilarMovie])
@limiter.limit("30/minute")
def similar(request: Request, movie_id: Annotated[int, Path(ge=1, le=1682)]):
    if movie_id not in data["similarity_df"].index:
        raise HTTPException(404, f"movie_id {movie_id} not found")

    candidates = (
        data["top_similar"][data["top_similar"]["movie_id"] == movie_id]
        .head(5)
        .merge(
            data["movies"].reset_index()[["movie_id", "title", "genres"]],
            left_on="similar_movie_id",
            right_on="movie_id"
        )[["similar_movie_id", "title", "genres", "similarity_score"]]
        .rename(columns={"similar_movie_id": "movie_id"})
    )
    return candidates.to_dict(orient="records")


# Recommend movies based on liked movie IDs using a hybrid of SVD similarity and genre profile matching
@app.post("/recommendations/from-liked-movies", response_model=list[Recommendation])
@limiter.limit("20/minute")
def recommend(request: Request, liked_movies: LikedMovies):
    sim = data["similarity_df"]
    movies = data["movies"]
    
    # Validate movie IDs
    not_found = [m for m in liked_movies.movie_ids if m not in sim.index]
    if not_found:
        raise HTTPException(404, f"These movie IDs were not found: {not_found}")

    profile = get_genre_profile(liked_movies.movie_ids)
    svd_scores = sim[liked_movies.movie_ids].mean(axis=1).drop(index=liked_movies.movie_ids, errors="ignore")
    gv = data["genre_vectors"]

    results = []
    for movie_id, svd_score in svd_scores.items():
        if movie_id not in movies.index: # Removed movies with 'unknown' genres
            continue
        g_score = float(np.dot(gv.loc[movie_id].values, profile)) if movie_id in gv.index else 0.0
        results.append({
            "movie_id":    int(movie_id),
            "title":       movies.loc[movie_id, "title"],
            "final_score": round(0.7 * float(svd_score) + 0.3 * g_score, 6),
            "genres":      movies.loc[movie_id, "genres"]
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)[:5]