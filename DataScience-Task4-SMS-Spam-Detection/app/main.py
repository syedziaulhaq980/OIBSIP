from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib

app = FastAPI(
    title="SMS Spam Detection API",
    description="API for detecting whether an SMS message is spam or ham.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "spam_detection_pipeline.pkl")


class SMSRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="SMS message to classify as spam or ham"
    )


@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is running"
    }


@app.post("/predict")
def predict_sms(request: SMSRequest):

    prediction = model.predict([request.message])[0]

    return {
        "message": request.message,
        "prediction": prediction
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded"
    }



