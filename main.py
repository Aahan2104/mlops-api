from fastapi.security import APIKeyHeader
from fastapi import Security
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pickle
import numpy as np
import logging
import time

# app initialization
app = FastAPI()

API_KEY = "mysecretkey123"
api_key_header = APIKeyHeader(name="X-API-Key")
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
@app.post("/predict")
def predict(data: IrisInput, api_key: str = Security(api_key_header)):
    verify_api_key(api_key)

# --------------------------
# Configure Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------
# Load Model & Scaler
# --------------------------
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    logger.info("Model and scaler loaded successfully!")

except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

# --------------------------
# Initialize FastAPI
# --------------------------
app = FastAPI()

# --------------------------
# Request Logging Middleware
# --------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(f"Incoming request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Completed in {process_time:.4f} sec")

    return response

# --------------------------
# Input Schema
# --------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --------------------------
# Home Route
# --------------------------
@app.get("/")
def home():
    return {"message": "API with Logging & Error Handling is running!"}

# --------------------------
# Prediction Endpoint
# --------------------------
@app.post("/predict")
def predict(data: IrisInput):
    try:
        logger.info(f"Received input: {data}")

        input_data = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        logger.info(f"Prediction: {prediction[0]}")

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during prediction"
        )