from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class SensorReading(Base):
    """Model for sensor readings from solar monitoring devices."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    voltage_v = Column(Float)
    current_a = Column(Float)
    temp_c = Column(Float)
    humidity_pct = Column(Float)
    lux = Column(Float)
    
    def __repr__(self):
        return f"<SensorReading(device_id='{self.device_id}', timestamp='{self.timestamp}')>"

class RelayControl(Base):
    """Model for relay control events."""
    __tablename__ = "relay_controls"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    relay_id = Column(String, index=True)
    action = Column(Boolean)  # True for ON, False for OFF
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<RelayControl(device_id='{self.device_id}', relay_id='{self.relay_id}', action='{self.action}')>"