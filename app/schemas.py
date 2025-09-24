from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SensorReadingBase(BaseModel):
    """Base schema for sensor readings."""
    device_id: str
    voltage_v: float
    current_a: float
    temp_c: float
    humidity_pct: float
    lux: float

class SensorReadingCreate(SensorReadingBase):
    """Schema for creating sensor readings."""
    ts: Optional[datetime] = Field(default_factory=datetime.utcnow)

class SensorReading(SensorReadingBase):
    """Schema for returning sensor readings."""
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

class RelayControlBase(BaseModel):
    """Base schema for relay control."""
    device_id: str
    relay_id: str
    action: bool  # True for ON, False for OFF

class RelayControlCreate(RelayControlBase):
    """Schema for creating relay control events."""
    pass

class RelayControl(RelayControlBase):
    """Schema for returning relay control events."""
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)