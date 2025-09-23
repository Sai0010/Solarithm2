import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from app.main import app
from app.db import Base, engine, get_db
from app.models import SensorReading, RelayControl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine_test = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

# Override the get_db dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Test client
client = TestClient(app)

@pytest.fixture(scope="function")
def test_db():
    # Create the tables
    Base.metadata.create_all(bind=engine_test)
    yield
    # Drop the tables after the test
    Base.metadata.drop_all(bind=engine_test)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data

def test_create_sensor_reading(test_db):
    """Test creating a sensor reading."""
    # Test data
    sensor_data = {
        "device_id": "test_device",
        "ts": datetime.utcnow().isoformat(),
        "voltage_v": 15.5,
        "current_a": 2.5,
        "temp_c": 25.0,
        "humidity_pct": 45.0,
        "lux": 800.0
    }
    
    # Post the data
    response = client.post("/api/sensors", json=sensor_data)
    assert response.status_code == 200
    data = response.json()
    
    # Check the response
    assert data["device_id"] == sensor_data["device_id"]
    assert data["voltage_v"] == sensor_data["voltage_v"]
    assert data["current_a"] == sensor_data["current_a"]
    assert data["temp_c"] == sensor_data["temp_c"]
    assert data["humidity_pct"] == sensor_data["humidity_pct"]
    assert data["lux"] == sensor_data["lux"]
    assert "id" in data
    assert "timestamp" in data

def test_get_latest_reading(test_db):
    """Test getting the latest sensor reading."""
    # First create a reading
    sensor_data = {
        "device_id": "test_device",
        "ts": datetime.utcnow().isoformat(),
        "voltage_v": 15.5,
        "current_a": 2.5,
        "temp_c": 25.0,
        "humidity_pct": 45.0,
        "lux": 800.0
    }
    client.post("/api/sensors", json=sensor_data)
    
    # Now get the latest reading
    response = client.get("/api/latest?device_id=test_device")
    assert response.status_code == 200
    data = response.json()
    
    # Check the response
    assert data["device_id"] == sensor_data["device_id"]
    assert data["voltage_v"] == sensor_data["voltage_v"]
    assert data["current_a"] == sensor_data["current_a"]
    assert data["temp_c"] == sensor_data["temp_c"]
    assert data["humidity_pct"] == sensor_data["humidity_pct"]
    assert data["lux"] == sensor_data["lux"]

def test_control_relay(test_db):
    """Test controlling a relay."""
    # Test data
    relay_data = {
        "device_id": "test_device",
        "relay_id": "test_relay",
        "action": True
    }
    
    # Post the data
    response = client.post("/api/control", json=relay_data)
    assert response.status_code == 200
    data = response.json()
    
    # Check the response
    assert data["device_id"] == relay_data["device_id"]
    assert data["relay_id"] == relay_data["relay_id"]
    assert data["action"] == relay_data["action"]
    assert "id" in data
    assert "timestamp" in data

def test_forecast(test_db):
    """Test the forecast endpoint."""
    # Create some readings first
    for i in range(24):
        sensor_data = {
            "device_id": "test_device",
            "ts": (datetime.utcnow().replace(hour=i)).isoformat(),
            "voltage_v": 15.5,
            "current_a": 2.5,
            "temp_c": 25.0,
            "humidity_pct": 45.0,
            "lux": 800.0
        }
        client.post("/api/sensors", json=sensor_data)
    
    # Now get the forecast
    response = client.get("/api/forecast?device_id=test_device&horizon=3")
    assert response.status_code == 200
    data = response.json()
    
    # Check the response
    assert data["device_id"] == "test_device"
    assert data["horizon"] == 3
    assert "forecast_generated" in data
    assert "forecast" in data
    assert len(data["forecast"]) == 3
    
    # Check each forecast point
    for point in data["forecast"]:
        assert "timestamp" in point
        assert "power" in point