import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

df = pd.read_csv("sales.csv")

X = df[["units_sold", "region", "product"]]
y = df["revenue"]

preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["region", "product"])
], remainder="passthrough")

model_v2 = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model_v2.fit(X, y)

joblib.dump(model_v2, "revenue_model_v2.pkl")
print("Improved model saved.")
