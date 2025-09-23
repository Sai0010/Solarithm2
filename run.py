import uvicorn
import logging
from app.init_db import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Start the FastAPI server
    logger.info("Starting SolArithm API server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)