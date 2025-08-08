from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np


class FlightRequest(BaseModel):
    Origin: str = Field(..., min_length=3, max_length=3)
    Dest: str = Field(..., min_length=3, max_length=3)
    FlightDate: str  # ISO date string
    CRSDepTime: int  # minutes since midnight
    CRSArrTime: int  # minutes since midnight
    temperature_c: Optional[float] = 0
    precip_mm: Optional[float] = 0
    cloud_pct: Optional[float] = 0
    wind_speed_mps: Optional[float] = 0


app = FastAPI(title="Flight AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"]
)


@app.get("/")
def root():
    return {"status": "ok", "name": "Flight AI API"}


@app.post("/predict")
def predict(req: FlightRequest):
    try:
        # Base delay
        base_delay = 5.0

        # Peak-hour contribution from departure time (minutes -> hour)
        hour_value = int(req.CRSDepTime) // 60
        if (7 <= hour_value <= 9) or (16 <= hour_value <= 18):
            base_delay += 10.0

        # Weather impacts
        temp = float(req.temperature_c or 0)
        if abs(temp - 20.0) > 15.0:
            base_delay += 5.0

        precip = float(req.precip_mm or 0)
        if precip > 5.0:
            base_delay += 8.0

        wind = float(req.wind_speed_mps or 0)
        if wind > 10.0:
            base_delay += 7.0

        # Hubs heuristic
        hubs = {"ORD", "DEN", "IAH", "EWR", "SFO", "LAX", "IAD"}
        if req.Origin.upper() in hubs or req.Dest.upper() in hubs:
            base_delay += 3.0
        if req.Origin.upper() in hubs and req.Dest.upper() in hubs:
            base_delay += 2.0

        # Final delay with small noise
        delay = max(0.0, base_delay + float(np.random.normal(0, 2)))

        return {
            "predicted_delay": round(float(delay), 2),
            "confidence": 0.85,
            "model_used": "Heuristic Demo",
            "status": "success",
        }
    except Exception as e:
        return {
            "predicted_delay": None,
            "confidence": 0,
            "model_used": "Heuristic Demo",
            "status": "error",
            "error": str(e),
        }


# For local run: uvicorn server.main:app --reload

