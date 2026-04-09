from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Secure FL Backend API")

# Allow CORS for potential frontend separation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process = None

class TrainingConfig(BaseModel):
    num_clients: int = 3
    num_rounds: int = 10
    privacy_multiplier: float = 0.5

@app.get("/")
def health_check():
    return {"status": "ok", "service": "FL Backend Controller"}

@app.post("/api/start_training")
def start_training(config: TrainingConfig):
    global process
    # Clear old metrics
    if os.path.exists('metrics.json'):
        try:
            os.remove('metrics.json')
        except Exception:
            pass
            
    # Launch simulator in background
    process = subprocess.Popen([
        "python", "simulator.py", 
        str(config.num_clients), 
        str(config.num_rounds), 
        str(config.privacy_multiplier)
    ])
    
    return {"message": "Training started successfully"}

@app.get("/api/status")
def get_status():
    global process
    if process and process.poll() is None:
        return {"status": "training"}
    return {"status": "idle"}

@app.get("/api/metrics")
def get_metrics():
    if os.path.exists('metrics.json'):
        try:
            with open('metrics.json', 'r') as f:
                data = json.load(f)
            return {"metrics": data}
        except json.JSONDecodeError:
            return {"metrics": []}
    return {"metrics": []}
