import numpy as np
import pandas as pd

## Datasets Link !!!
## https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset

movie = pd.read_csv('../input/movielens-20m-dataset/movie.csv')
rating = pd.read_csv('../input/movielens-20m-dataset/rating.csv')
df = movie.merge(rating,how="left",on="movieId")
df.head()

yorum_sayıları = pd.DataFrame(df["title"].value_counts())
cıkacaklar = yorum_sayıları[yorum_sayıları["title"]<=1000].index
print(cıkacaklar)
common_movies = df[~df["title"].isin(cıkacaklar)]

user_movie_df =common_movies.pivot_table(index=["userId"],columns=["title"],values = "rating")
rasgele_kullanıcı = int(pd.Series(user_movie_df.index).sample(1,random_state = 120).values)
random_user_df = user_movie_df[user_movie_df.index == rasgele_kullanıcı]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
print(len(movies_watched))

movies_watched_df = user_movie_df[movies_watched]

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count=user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]

users_same_movies = user_movie_count[user_movie_count["movie_count"] > ((len(movies_watched)*60 / 100))]

final_df = movies_watched_df.iloc[users_same_movies.index]

corr_df = final_df.T.corr().unstack().sort_values(ascending=False).drop_duplicates()
corr_df = pd.DataFrame(corr_df,columns = ["corr"])
corr_df.index.names = ["userId_1","userId_2"]
corr_df = corr_df.reset_index()

corr_df[(corr_df["userId_1"]==rasgele_kullanıcı)& (corr_df["corr"]>=0.65)]

top_users = corr_df[(corr_df["userId_1"] == rasgele_kullanıcı) & (corr_df["corr"] >= 0.65)][
    ["userId_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"userId_2": "userId"}, inplace=True)

rating = pd.read_csv('../input/movielens-20m-dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != rasgele_kullanıcı]

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df=recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values(by="weighted_rating",ascending=False)

movie = pd.read_csv('../input/movielens-20m-dataset/movie.csv')
recommendation_df.merge(movie[["movieId", "title"]])

x = df[(df["userId"] == 107674.0)&(df["rating"]== 5.0)]
x[(x["timestamp"] == x["timestamp"].max())]["title"]

movie_name = "Royal Tenenbaums, The (2001)"
movie_name = user_movie_df[movie_name]

user_movie_df.corrwith(movie_name).sort_values(ascending=False)

önerilen_ilk_5_film = user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:6]