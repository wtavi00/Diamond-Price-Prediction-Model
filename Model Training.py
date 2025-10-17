# EDA and Model Training.ipynb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("data/diamonds.csv")

# Encode categorical columns
for col in ["cut", "color", "clarity"]:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

joblib.dump(model, "app/models/model.pkl")
print("✅ Model saved to app/models/model.pkl")
