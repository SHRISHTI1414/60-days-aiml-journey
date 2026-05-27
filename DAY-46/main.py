from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():

    return {
        "message": "AI Model API Running Successfully"
    }

@app.post("/predict")
def predict(data: PredictionRequest):

    start_time = time.time()

    text = data.text.lower()

    if "ai" in text:
        prediction = "AI Related Query"

    elif "python" in text:
        prediction = "Programming Query"

    elif "machine learning" in text:
        prediction = "ML Related Query"

    else:
        prediction = "General Query"

    latency = round(time.time() - start_time, 4)

    return {
        "input": data.text,
        "prediction": prediction,
        "latency_seconds": latency
    }

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )