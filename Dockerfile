FROM python:3.12-slim

WORKDIR /movie_api

RUN pip install uv

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --frozen

COPY . .