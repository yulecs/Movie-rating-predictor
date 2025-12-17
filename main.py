import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

ratings.head()

movie_cols = [
    "movieId", "title", "release_date", "video_release_date",
    "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    names=movie_cols,
    encoding="latin-1"
)

movies.head()


movie_ratings = ratings.groupby("movieId").agg(
    avg_rating=("rating", "mean"), 
    num_ratings=("rating", "count")
)

data = movies.merge(movie_ratings, on="movieId")

genre_cols = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

data["year"] = data["title"].str.extract(r"\((\d{4})\)")
data["year"] = data["year"].astype(float)

X = data[genre_cols + ["num_ratings"]]

y = data["avg_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()

model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = root_mean_squared_error(y_test, preds)
print(rmse)


