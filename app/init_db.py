import logging
from datetime import datetime, timedelta
import random
from sqlalchemy.orm import Session

from app.db import engine, SessionLocal, Base
from app.models import SensorReading, RelayControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize database tables and create sample data."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
    
    # Create a session
    db = SessionLocal()
    
    try:
        # Check if we already have data
        if db.query(SensorReading).count() == 0:
            logger.info("Adding sample sensor readings...")
            _create_sample_sensor_data(db)
            logger.info("Sample sensor readings added")
            
        if db.query(RelayControl).count() == 0:
            logger.info("Adding sample relay control data...")
            _create_sample_relay_data(db)
            logger.info("Sample relay control data added")
            
        db.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def _create_sample_sensor_data(db: Session):
    """Create sample sensor readings for the past 24 hours."""
    device_id = "solar_panel_01"
    now = datetime.utcnow()
    
    # Create readings for the past 24 hours, one per hour
    for hour in range(24, 0, -1):
        timestamp = now - timedelta(hours=hour)
        
        # Simulate a day cycle with peak at noon
        hour_of_day = timestamp.hour
        sun_factor = max(0, 1 - abs(hour_of_day - 12) / 12)
        
        # Create sample reading with realistic values
        reading = SensorReading(
            device_id=device_id,
            timestamp=timestamp,
            voltage_v=12.0 + (sun_factor * 6.0) + random.uniform(-0.5, 0.5),  # 12-18V
            current_a=0.1 + (sun_factor * 4.9) + random.uniform(-0.1, 0.1),   # 0.1-5A
            temp_c=20.0 + (sun_factor * 15.0) + random.uniform(-2.0, 2.0),    # 20-35°C
            humidity_pct=50.0 + random.uniform(-10.0, 10.0),                  # 40-60%
            lux=100 + (sun_factor * 900) + random.uniform(-50, 50)            # 100-1000 lux
        )
        db.add(reading)
    
    db.commit()

def _create_sample_relay_events(db: Session):
    """Create sample relay control events."""
    device_id = "solar_panel_01"
    now = datetime.utcnow()
    
    # Create a few relay events
    events = [
        RelayControl(
            device_id=device_id,
            relay_id="battery_charger",
            action=True,  # ON
            timestamp=now - timedelta(hours=12)
        ),
        RelayControl(
            device_id=device_id,
            relay_id="battery_charger",
            action=False,  # OFF
            timestamp=now - timedelta(hours=6)
        ),
        RelayControl(
            device_id=device_id,
            relay_id="inverter",
            action=True,  # ON
            timestamp=now - timedelta(hours=8)
        ),
        RelayControl(
            device_id=device_id,
            relay_id="inverter",
            action=False,  # OFF
            timestamp=now - timedelta(hours=4)
        )
    ]
    
    for event in events:
        db.add(event)
    
    db.commit()

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialization completed")