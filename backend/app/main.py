# File: backend/app/main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import logging
import traceback
from app.services.logging_config import setup_logging

# Setup logging
log_file = setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Database imports
from app.database.db import engine
from app.database import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

from app.api import health
from app.api import sheets
from app.api import sheets_headers          # Import the sheets headers API
from app.api import schema
from app.api import schema_save
from app.api import schema_superscout, schema_superscout_save
from app.api import test_unified
from app.api import validate
from app.api import setup
from app.api import prompt_builder_router
from app.api import unified_dataset  # Make sure this import is here
from app.api import field_selection  # Import the new field selection router
from app.api import picklist_analysis
from app.api import picklist_generator
from app.api import test_schema_superscout  # Import the test endpoint
from app.api import test_enhanced_parser    # Import the enhanced parser test
from app.api import progress                # Import the progress tracking API
from app.api import debug_logs              # Import debug logs API
from app.api import alliance_selection      # Import alliance selection API
from app.api import archive                 # Import event archiving API
from app.api import sheet_config            # Import sheet configuration API
from app.api import picklist_comparison     # Import picklist comparison API
from app.api import manual_search           # Import manual search API

app = FastAPI(title="FRC Scouting Assistant", version="0.1.0")

# Allow frontend (localhost:5173) to call backend (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to log all unhandled exceptions
    """
    error_msg = f"Unhandled exception: {str(exc)}"
    stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Log the full error with request details
    logger.error(f"{error_msg}\nRequest: {request.method} {request.url}\n{stack_trace}")

    # Return a JSON response
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )

# Include routers
app.include_router(health.router, prefix="/api/health")
app.include_router(sheets.router, prefix="/api/sheets")
app.include_router(sheets_headers.router)  # Already has prefix="/api/sheets" in its definition
app.include_router(schema.router, prefix="/api/schema")
app.include_router(schema_save.router, prefix="/api/schema")
app.include_router(schema_superscout.router, prefix="/api/schema/super")
app.include_router(schema_superscout_save.router, prefix="/api/schema/super")
app.include_router(test_unified.router)
app.include_router(validate.router, prefix="/api")
app.include_router(setup.router)
app.include_router(prompt_builder_router.router, prefix="/api")
app.include_router(unified_dataset.router, prefix="/api/unified")
app.include_router(field_selection.router)  # Add the field selection router
app.include_router(picklist_analysis.router, prefix="/api")
app.include_router(picklist_generator.router)
app.include_router(test_schema_superscout.router)  # Add the test endpoint
app.include_router(test_enhanced_parser.router)     # Add the enhanced parser test
app.include_router(progress.router, prefix="/api")  # Add the progress tracking API
app.include_router(debug_logs.router, prefix="/api/debug")  # Add the debug logs API
app.include_router(alliance_selection.router)  # Add the alliance selection API
app.include_router(archive.router)  # Add the event archiving API
app.include_router(picklist_comparison.router)  # Add the picklist comparison API
app.include_router(sheet_config.router)  # Sheet configuration API already has prefix
app.include_router(manual_search.router)  # Add the manual search API

@app.get("/")
async def root():
    return {"message": "Backend is running!"}