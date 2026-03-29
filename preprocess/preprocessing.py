import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from math import log


# Ratings
ratings = pd.read_csv(
    "../data/raw/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Genres
genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Movies
movies = pd.read_csv(
    "../data/raw/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1] + list(range(5, 24))
)
movies.columns = ["movie_id", "title"] + genre_cols # rename the relevant columns


# ── Movie Metadata ─────────────────────────────────────
def extract_genres(row):
    return [genre for genre in genre_cols if row[genre] == 1]

movies["genre_list"] = movies.apply(extract_genres, axis=1)
movies["genres"]     = movies["genre_list"].apply(lambda x: "|".join(x))

# Remove 'unknown' genre tag
movies["genre_list"] = movies["genre_list"].apply(
    lambda x: [g for g in x if g != "unknown"]
)

movies = movies[movies["genre_list"].apply(len) > 0].reset_index(drop=True)

# ── Remove movies with 'unknown' genres
valid_movie_ids = set(movies["movie_id"])
ratings = ratings[ratings["movie_id"].isin(valid_movie_ids)]
print(f"Ratings after filtering: {len(ratings)}")

movies_metadata = movies[["movie_id", "title", "genres"]].copy()
movies_metadata.to_csv("../data/movies_metadata.csv", index=False)
print("Movie metadata saved to movies_metadata.csv")


# ── Similarity Matrix ─────────────────────────────────────
user_item = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
)

# ── Mean Centering 
user_means = user_item.mean(axis=1)
user_item_centered = user_item.sub(user_means, axis=0).fillna(0)
movie_user = user_item_centered.T  # transpose to get MxU 

# ── SVD
k = 20
U, sigma, Vt = svds(user_item_centered.values, k=k)

# svds returns smallest singular values first (reverse so largest is first)
U     = U[:, ::-1]
sigma = sigma[::-1]
Vt    = Vt[::-1, :]

# Movie latent matrix — shape (n_movies × k)
movie_latent = Vt.T

# ── Cosine Similarity 
similarity = cosine_similarity(movie_latent)
similarity_matrix = pd.DataFrame(
    similarity,
    index=movie_user.index,
    columns=movie_user.index
)
similarity_matrix.to_csv("../data/similarity_matrix.csv", index=True)
print("Similarity matrix saved to similarity_matrix.csv")


# ── Top 20 Similar Movies ─────────────────────────────────────
TOP_N = 20

rows = []

for movie_id in similarity_matrix.index:
    sims = similarity_matrix.loc[movie_id].drop(movie_id)
    top_n = sims.sort_values(ascending=False).head(TOP_N)

    for rank, (similar_movie_id, score) in enumerate(top_n.items(), start=1):
        rows.append({
            "movie_id": int(movie_id),
            "similar_movie_id": int(similar_movie_id),
            "rank": rank,
            "similarity_score": round(float(score), 6)
        })

top_similar_movies = pd.DataFrame(rows)
top_similar_movies.to_csv("../data/top_similar_movies.csv", index=False)
print("Top similar movies saved to top_similar_movies.csv")


# ── Inverse Document Frequency for Genres ─────────────────────────────────────
genres_exploded = movies["genre_list"].explode()
genres_exploded = genres_exploded[genres_exploded != "unknown"]

genre_counts = genres_exploded.value_counts()
total_movies = len(movies)

genre_idf = {genre: log(total_movies / count) for genre, count in genre_counts.items()}


# ── Genre IDF
all_genres = list(genre_idf.keys())

rows = []
for _, row in movies.iterrows():
    vec = {"movie_id": row["movie_id"]}
    for genre in all_genres:
        vec[genre] = genre_idf[genre] if genre in row["genre_list"] else 0.0
    rows.append(vec)

movie_genre_vectors = pd.DataFrame(rows).set_index("movie_id")
movie_genre_vectors.to_csv("../data/movie_genre_vectors.csv")
print("Movie genre vectors saved to movie_genre_vectors.csv")