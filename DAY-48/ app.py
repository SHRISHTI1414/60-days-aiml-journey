from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/")
async def home():

    return {
        "message": "ASGI FastAPI Server Running"
    }

@app.get("/status")
async def status():

    current_time = time.time()

    return {
        "server_status": "active",
        "timestamp": current_time
    }

@app.get("/ai-response")
async def ai_response(query: str):

    return {
        "query": query,
        "response": f"AI processed your query: {query}"
    }