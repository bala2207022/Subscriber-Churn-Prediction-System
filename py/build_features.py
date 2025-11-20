import pandas as pd
from datetime import timedelta

users   = pd.read_csv("/Users/balakrishna/Documents/bala_py_db/netflix/data/users.csv")
logins  = pd.read_csv("/Users/balakrishna/Documents/bala_py_db/netflix/data/logins.csv")
watch   = pd.read_csv("/Users/balakrishna/Documents/bala_py_db/netflix/data/watch.csv")
ratings = pd.read_csv("/Users/balakrishna/Documents/bala_py_db/netflix/data/ratings.csv")
shows   = pd.read_csv("/Users/balakrishna/Documents/bala_py_db/netflix/data/tv.csv")   

logins["login_date"]  = pd.to_datetime(logins["login_date"])
watch["watch_date"]   = pd.to_datetime(watch["watch_date"])
users["signup_date"]  = pd.to_datetime(users["signup_date"])

max_date = max(logins["login_date"].max(), watch["watch_date"].max())

last_login = (
    logins
    .groupby("user_id", as_index=False)["login_date"]
    .max()
)
last_login["days_since_last_login"] = (max_date - last_login["login_date"]).dt.days
feat_last_login = last_login[["user_id", "days_since_last_login"]]

cutoff_30d = max_date - timedelta(days=30)
recent_watch_30 = watch[watch["watch_date"] >= cutoff_30d]

watch_30 = (
    recent_watch_30
    .groupby("user_id", as_index=False)["watch_time"]
    .sum()
    .rename(columns={"watch_time": "total_watch_time_30d"})
)

rating_stats = (
    ratings
    .groupby("user_id")["rating"]
    .agg(avg_rating="mean", num_ratings="count")
    .reset_index()
)

users["account_age_days"] = (max_date - users["signup_date"]).dt.days
feat_account_age = users[["user_id", "age", "plan_type", "account_age_days", "churned"]]

features = (
    feat_account_age
    .merge(feat_last_login, on="user_id", how="left")
    .merge(watch_30,        on="user_id", how="left")
    .merge(rating_stats,    on="user_id", how="left")
)

features["days_since_last_login"] = features["days_since_last_login"].fillna(999)
features["total_watch_time_30d"]  = features["total_watch_time_30d"].fillna(0)
features["avg_rating"]            = features["avg_rating"].fillna(0)
features["num_ratings"]           = features["num_ratings"].fillna(0)

print(features.head())
features.to_csv("churn_features.csv", index=False)
print("Saved churn_features.csv with shape:", features.shape)
