from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "AI Prediction API is Running"
    }

@app.post("/predict")
def predict(data: PredictionRequest):

    text = data.text.lower()

    if "ai" in text:
        prediction = "AI Related Query"

    elif "python" in text:
        prediction = "Programming Query"

    else:
        prediction = "General Query"

    return {
        "input": data.text,
        "prediction": prediction
    }

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )