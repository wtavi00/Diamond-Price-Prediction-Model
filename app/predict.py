import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

cut_encoder = LabelEncoder().fit(["Fair", "Good", "Very Good", "Premium", "Ideal"])
color_encoder = LabelEncoder().fit(["J", "I", "H", "G", "F", "E", "D"])
clarity_encoder = LabelEncoder().fit(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

model = joblib.load("app/models/model.pkl")

def predict_price(data):
    features = np.array([[
        data.carat,
        cut_encoder.transform([data.cut])[0],
        color_encoder.transform([data.color])[0],
        clarity_encoder.transform([data.clarity])[0],
        data.depth,
        data.table,
        data.x,
        data.y,
        data.z
    ]])
    price = model.predict(features)[0]
    return round(float(price), 2)
