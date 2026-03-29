from typing import Annotated
from pydantic import BaseModel, Field

# Pydantic models for data validation 
class LikedMovies(BaseModel):
    movie_ids: Annotated[list[int], Field(min_length=2, max_length=5)]


class MovieResult(BaseModel):
    movie_id: int
    title: str
    genres: str


class SimilarMovie(BaseModel):
    movie_id: int
    title: str
    similarity_score: float
    genres: str


class Recommendation(BaseModel):
    movie_id: int
    title: str
    final_score: float
    genres: str