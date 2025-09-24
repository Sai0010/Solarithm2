import logging
from datetime import datetime, timedelta
import random
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.db import engine, SessionLocal, Base
from app.models import SensorReading, RelayControl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables."""
    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def insert_sample_data():
    """Insert sample data into tables if they are empty."""
    db = SessionLocal()
    try:
        # Check and insert sensor readings
        if db.query(SensorReading).count() == 0:
            logger.info("Adding sample sensor readings...")
            _create_sample_sensor_data(db)
            logger.info("Sample sensor readings added")

        # Check and insert relay control data
        if db.query(RelayControl).count() == 0:
            logger.info("Adding sample relay control data...")
            _create_sample_relay_data(db)
            logger.info("Sample relay control data added")

        db.commit()
    except OperationalError as e:
        logger.error(f"Database operation failed. Make sure tables are created: {e}")
        db.rollback()
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def _create_sample_sensor_data(db: Session):
    """Create sample sensor readings for the past 24 hours."""
    device_id = "solar_panel_01"
    now = datetime.utcnow()
    
    for hour in range(24, 0, -1):
        timestamp = now - timedelta(hours=hour)
        hour_of_day = timestamp.hour
        sun_factor = max(0, 1 - abs(hour_of_day - 12) / 12)
        
        reading = SensorReading(
            device_id=device_id,
            timestamp=timestamp,
            voltage_v=12.0 + (sun_factor * 6.0) + random.uniform(-0.5, 0.5),
            current_a=0.1 + (sun_factor * 4.9) + random.uniform(-0.1, 0.1),
            temp_c=20.0 + (sun_factor * 15.0) + random.uniform(-2.0, 2.0),
            humidity_pct=50.0 + random.uniform(-10.0, 10.0),
            lux=100 + (sun_factor * 900) + random.uniform(-50, 50)
        )
        db.add(reading)
    
    # Commit is not necessary here, it's handled in the calling function `insert_sample_data`

def _create_sample_relay_data(db: Session):
    """Create sample relay control events."""
    device_id = "solar_panel_01"
    now = datetime.utcnow()
    
    events = [
        RelayControl(device_id=device_id, relay_id="battery_charger", action=True, timestamp=now - timedelta(hours=12)),
        RelayControl(device_id=device_id, relay_id="battery_charger", action=False, timestamp=now - timedelta(hours=6)),
        RelayControl(device_id=device_id, relay_id="inverter", action=True, timestamp=now - timedelta(hours=8)),
        RelayControl(device_id=device_id, relay_id="inverter", action=False, timestamp=now - timedelta(hours=4))
    ]
    
    for event in events:
        db.add(event)
    
    # Commit is not necessary here, it's handled in the calling function `insert_sample_data`

if __name__ == "__main__":
    logger.info("Initializing database...")
    create_tables()
    insert_sample_data()
    logger.info("Database initialization completed")