import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db, engine
from app.models import Base, SensorReading, RelayControl
from app.schemas import (
    SensorReadingCreate, 
    SensorReading as SensorReadingSchema,
    RelayControlCreate,
    RelayControl as RelayControlSchema,
    HealthResponse
)
from app.ml_client import MLClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SolArithm API",
    description="AI-powered Solar Monitoring & Optimization API",
    version="0.1.0",
)

# Initialize ML client
ml_client = MLClient()

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting SolArithm API")
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.utcnow()
    }

@app.post("/api/sensors", response_model=SensorReadingSchema)
async def create_sensor_reading(
    reading: SensorReadingCreate,
    db: Session = Depends(get_db)
):
    """
    Store sensor reading in the database.
    
    Accepts JSON payload with device_id, timestamp, and sensor values.
    """
    try:
        # Create DB model from schema
        db_reading = SensorReading(
            device_id=reading.device_id,
            timestamp=reading.ts,
            voltage_v=reading.voltage_v,
            current_a=reading.current_a,
            temp_c=reading.temp_c,
            humidity_pct=reading.humidity_pct,
            lux=reading.lux
        )
        
        # Add to DB and commit
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)
        
        logger.info(f"Stored sensor reading for device {reading.device_id}")
        return db_reading
    except Exception as e:
        logger.error(f"Error storing sensor reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing sensor reading: {str(e)}")

@app.get("/api/latest", response_model=SensorReadingSchema)
async def get_latest_reading(
    device_id: str = Query(..., description="Device ID to get latest reading for"),
    db: Session = Depends(get_db)
):
    """
    Get the latest sensor reading for a specific device.
    """
    try:
        # Query the latest reading for the device
        reading = db.query(SensorReading).filter(
            SensorReading.device_id == device_id
        ).order_by(SensorReading.timestamp.desc()).first()
        
        if not reading:
            raise HTTPException(status_code=404, detail=f"No readings found for device {device_id}")
        
        return reading
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving latest reading: {str(e)}")

@app.get("/api/forecast")
async def get_forecast(
    device_id: str = Query(..., description="Device ID to forecast for"),
    horizon: int = Query(6, description="Number of hours to forecast"),
    db: Session = Depends(get_db)
):
    """
    Get power forecast for a specific device.
    
    Uses ML model for prediction or falls back to baseline if model not available.
    """
    try:
        # Get recent readings for the device (last 24 hours)
        readings = db.query(SensorReading).filter(
            SensorReading.device_id == device_id
        ).order_by(SensorReading.timestamp.desc()).limit(24).all()
        
        if not readings:
            raise HTTPException(status_code=404, detail=f"No readings found for device {device_id}")
        
        # Convert to DataFrame for ML model
        df = pd.DataFrame([{
            'timestamp': r.timestamp,
            'voltage_v': r.voltage_v,
            'current_a': r.current_a,
            'temp_c': r.temp_c,
            'humidity_pct': r.humidity_pct,
            'lux': r.lux,
            'power': r.voltage_v * r.current_a
        } for r in readings])
        
        # Get forecast from ML client
        forecast = ml_client.predict(df, horizon=horizon)
        
        # Create response with timestamps
        now = datetime.utcnow()
        forecast_data = [
            {
                "timestamp": now + pd.Timedelta(hours=i),
                "power": float(power)
            }
            for i, power in enumerate(forecast)
        ]
        
        return {
            "device_id": device_id,
            "forecast_generated": now,
            "horizon": horizon,
            "forecast": forecast_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.post("/api/control", response_model=RelayControlSchema)
async def control_relay(
    control: RelayControlCreate,
    db: Session = Depends(get_db)
):
    """
    Control a relay for a specific device.
    
    Accepts JSON payload with device_id, relay_id, and action.
    """
    try:
        # Create DB model from schema
        db_control = RelayControl(
            device_id=control.device_id,
            relay_id=control.relay_id,
            action=control.action,
            timestamp=datetime.utcnow()
        )
        
        # Add to DB and commit
        db.add(db_control)
        db.commit()
        db.refresh(db_control)
        
        logger.info(f"Relay control: device={control.device_id}, relay={control.relay_id}, action={'ON' if control.action else 'OFF'}")
        return db_control
    except Exception as e:
        logger.error(f"Error controlling relay: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error controlling relay: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)