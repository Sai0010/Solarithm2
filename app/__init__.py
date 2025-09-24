# SolArithm Backend Package
from fastapi import FastAPI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Import routes after app is created to avoid circular imports
def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SolArithm API",
        description="AI-powered Solar Monitoring & Optimization API",
        version="0.1.0",
    )
    
    # Import and include routers
    from app.main import router as main_router
    app.include_router(main_router)
    
    return app