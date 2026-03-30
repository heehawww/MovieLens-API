import streamlit as st
import requests
import pandas as pd
import os

# API = "http://127.0.0.1:8000"
API = os.getenv("API_URL", "http://localhost:8000")

st.title("Movie Recommendation System")


tab1, tab2, tab3 = st.tabs(["Search Movies", "Similar Movies", "Recommendations"])


# ── Search ─────────────────────────────────────
with tab1:
    st.subheader("Search Movies")
    st.caption("Search by title or part of a title to find matching movies and their IDs.")
    query = st.text_input("Movie title")

    if query:
        res = requests.get(f"{API}/movies/search", params={"query": query})
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            df.columns = ["Movie ID", "Title", "Genres"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif res.status_code == 429:
            st.error("Rate limit exceeded. Please try again shortly.")
        elif res.status_code == 404:
            st.error(f"No movies found matching '{query}'.")


# ── Similar Movies ──────────────────────────────
with tab2:
    st.subheader("Find Similar Movies")
    st.caption("Returns the top 5 most similar movies based on SVD cosine similarity.")
    movie_id = st.number_input("Movie ID", min_value=1, max_value=1682, step=1)

    if st.button("Find Similar"):
        res = requests.get(f"{API}/movies/{int(movie_id)}/similar")
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            df.columns = ["Movie ID", "Title", "Similarity Score", "Genres"]
            df["Similarity Score"] = df["Similarity Score"].round(4)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif res.status_code == 404:
            st.error(f"Movie ID {movie_id} was not found. It may have been removed during preprocessing.")
        elif res.status_code == 429:
            st.error("Rate limit exceeded. Please try again shortly.")
        else:
            st.error("Something went wrong.")


# ── Recommendations ─────────────────────────────
with tab3:
    st.subheader("Blended Recommendations")
    st.caption("Enter 2–5 unique movie IDs to get personalised recommendations based on SVD similarity and genre profile matching.")
    liked_input = st.text_input("Movie IDs (e.g. 1, 50, 172)")

    if st.button("Get Recommendations"):
        try:
            liked_ids = list({int(x.strip()) for x in liked_input.split(",")}) # Use a set to deduplicate entries

            if not (2 <= len(liked_ids) <= 5):
                st.error("Please enter between 2 and 5 unique movie IDs.")
            else:
                res = requests.post(
                    f"{API}/recommendations/from-liked-movies",
                    json={"movie_ids": liked_ids}
                )
                if res.status_code == 200:
                    df = pd.DataFrame(res.json())
                    df.columns = ["Movie ID", "Title", "Final Score", "Genres"]
                    df["Final Score"] = df["Final Score"].round(4)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                elif res.status_code == 404:
                    st.error(res.json().get("detail", "Some movie IDs were not found."))

                elif res.status_code == 429:
                    st.error("Rate limit exceeded. Please try again shortly.")

                elif res.status_code == 422:
                    detail = res.json().get("detail", "Invalid input.")
                    if isinstance(detail, list):
                        st.error(detail[0].get("msg", "Invalid input."))
                    else:
                        st.error(detail)
                else:
                    st.error("Something went wrong.")
        except ValueError:
            st.error("Please enter valid numeric movie IDs separated by commas.")