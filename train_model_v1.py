import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("sales.csv")

X = df[["units_sold"]]
y = df["revenue"]

model_v1 = LinearRegression()
model_v1.fit(X, y)

joblib.dump(model_v1, "revenue_model_v1.pkl")
print("Baseline model saved.")
