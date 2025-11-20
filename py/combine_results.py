import pandas as pd

pred = pd.read_csv("predictions.csv")
users = pd.read_csv("/Users/balakrishna/Documents/bala_py_db/netflix/data/users.csv")
final = users.merge(pred, on="user_id", how="left")
final.to_csv("final_user_risk.csv", index=False)
print("Saved final_user_risk.csv")
print(final.head())
