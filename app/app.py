from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

from .predict import predict_price, predict_from_csv
from .schemas import DiamondFeatures

app = FastAPI(
    title="Diamond Price Prediction API",
    description="Predict diamond prices from features or CSV uploads",
    version="1.0"
)

# Enable CORS (if you want to call from a frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Single Diamond Prediction
# ---------------------------
@app.post("/predict")
def predict_single(data: DiamondFeatures):
    try:
        price = predict_price(data)
        return {"predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------
# CSV File Prediction
# ---------------------------
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        df_results = predict_from_csv(temp_path, save_results=False)
        # Convert to list of dicts for API response
        result = df_results.to_dict(orient="records")
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))

    # Remove temporary file
    os.remove(temp_path)
    return {"predictions": result}


# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Diamond Price Prediction API! Use /predict or /predict_csv."}
