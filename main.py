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

feature_cols = genre_cols + ["year"]
X = data[feature_cols]
y = data["avg_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()

model.fit(X_train, y_train)

def predict_movie_rating(genres, year):
    # Start with all-zero features
    input_data = {col: 0 for col in feature_cols}
    
    # Set genres
    for genre in genres:
        if genre in input_data:
            input_data[genre] = 1
    
    # Set year
    input_data["year"] = year
    
    # Convert to DataFrame (IMPORTANT)
    X_new = pd.DataFrame([input_data])
    
    # Predict
    predicted_rating = model.predict(X_new)[0]
    
    return predicted_rating

pred = predict_movie_rating(
    genres=["Romance", "Comedy", "Sci-Fi", "Animation", "Fantasy"],
    year=2022
)

print(f"Predicted rating: {pred:.2f}")



