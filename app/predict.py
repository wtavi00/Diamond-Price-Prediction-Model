import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "app/models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model file not found at '{MODEL_PATH}'. Please run Model Training.py first.")

model = joblib.load(MODEL_PATH)

# ---------------------------
# Setup Label Encoders (consistent with training)
# ---------------------------
cut_encoder = LabelEncoder().fit(["Fair", "Good", "Very Good", "Premium", "Ideal"])
color_encoder = LabelEncoder().fit(["J", "I", "H", "G", "F", "E", "D"])
clarity_encoder = LabelEncoder().fit(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])


# ---------------------------
# Predict single diamond price
# ---------------------------
def predict_price(data):
    """Predict price for a single diamond (object or dict with attributes)."""
    if isinstance(data, dict):
        carat = data["carat"]
        cut = data["cut"]
        color = data["color"]
        clarity = data["clarity"]
        depth = data["depth"]
        table = data["table"]
        x = data["x"]
        y = data["y"]
        z = data["z"]
    else:
        carat = data.carat
        cut = data.cut
        color = data.color
        clarity = data.clarity
        depth = data.depth
        table = data.table
        x = data.x
        y = data.y
        z = data.z

    features = np.array([[
        carat,
        cut_encoder.transform([cut])[0],
        color_encoder.transform([color])[0],
        clarity_encoder.transform([clarity])[0],
        depth,
        table,
        x,
        y,
        z
    ]])

    price = model.predict(features)[0]
    return round(float(price), 2)


# ---------------------------
# Predict from CSV file
# ---------------------------
def predict_from_csv(csv_path, save_results=True):
    """
    Predict diamond prices from a CSV file.
    Expects columns: carat, cut, color, clarity, depth, table, x, y, z
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)

    required_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Encode categorical features
    df["cut"] = cut_encoder.transform(df["cut"])
    df["color"] = color_encoder.transform(df["color"])
    df["clarity"] = clarity_encoder.transform(df["clarity"])

    # Predict
    features = df[required_cols]
    preds = model.predict(features)
    df["predicted_price"] = np.round(preds, 2)

    if save_results:
        output_path = os.path.splitext(csv_path)[0] + "_predicted.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")

    return df


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Single prediction example
    example_data = {
        "carat": 0.5,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS2",
        "depth": 61.5,
        "table": 55,
        "x": 5.0,
        "y": 5.1,
        "z": 3.2
    }

    price = predict_price(example_data)
    print(f"💎 Predicted single diamond price: {price}")

