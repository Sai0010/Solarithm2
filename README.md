# SolArithm — AI-powered Solar Monitoring & Optimization

SolArithm is a comprehensive solution for monitoring and optimizing solar panel systems using IoT sensors, machine learning, and a modern web interface.

## Project Structure

```
- app/           (backend FastAPI)
- ml_model/      (ML training + inference)
- scripts/       (raspi scripts)
- config.py      (configuration settings)
- requirements.txt (dependencies)
- run.py         (main entry point)
```

## Backend Setup (Task 1 Complete)

The backend is built with FastAPI and provides the following endpoints:

- `POST /api/sensors` - Store sensor readings
- `GET /api/latest` - Get latest sensor reading for a device
- `GET /api/forecast` - Get power forecast for a device
- `POST /api/control` - Control device relays
- `GET /api/health` - Service health check

### Setup Instructions

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python run.py
   ```
   
   Or directly with uvicorn:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Database

The application uses SQLite by default but is designed to be easily switched to PostgreSQL for production. The database models include:

- `SensorReading` - Stores sensor data (voltage, current, temperature, etc.)
- `RelayControl` - Logs relay control events

### ML Integration

The backend includes an ML client that:
- Uses a trained LSTM model for power forecasting when available
- Falls back to a baseline moving average model when the LSTM model is not available

## Next Steps

The following tasks are planned for future implementation:

- Task 2: ML Pipeline Skeleton
- Task 3: Raspberry Pi Scripts
- Task 4: Frontend Skeleton (React)
- Task 5: Integration & Tests
- Task 6: Complete Documentation