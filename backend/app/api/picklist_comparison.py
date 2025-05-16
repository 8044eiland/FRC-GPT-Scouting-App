# backend/app/api/picklist_comparison.py

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.services.picklist_comparison_service import PicklistComparisonService

router = APIRouter(prefix="/api/picklist", tags=["Picklist"])

class ChatMessage(BaseModel):
    role: str
    content: str

class MetricPriority(BaseModel):
    id: str
    weight: float = Field(1.0, ge=0.5, le=3.0)
    reason: Optional[str] = None

class ComparisonRequest(BaseModel):
    unified_dataset_path: str
    team_numbers: List[int]
    chat_history: Optional[List[ChatMessage]] = []
    specific_question: Optional[str] = None
    metrics_to_highlight: Optional[List[str]] = None
    priorities: Optional[List[MetricPriority]] = None

@router.post("/compare")
async def compare_teams(request: ComparisonRequest):
    """
    Compare two teams from the picklist and provide detailed analysis.
    
    Args:
        request: Comparison request with team numbers and optional specific question
        
    Returns:
        Detailed comparison data and analysis
    """
    try:
        # Validate inputs
        if len(request.team_numbers) != 2:
            raise HTTPException(status_code=400, detail="Exactly two team numbers must be provided for comparison")
        
        # Add detailed logging
        import logging
        logger = logging.getLogger("picklist_comparison")
        logger.info(f"Comparison request: {request.dict()}")
        
        try:
            # Initialize the service
            comparison_service = PicklistComparisonService(request.unified_dataset_path)
            
            # Process the comparison
            result = await comparison_service.compare_teams(
                team_numbers=request.team_numbers,
                chat_history=request.chat_history,
                specific_question=request.specific_question,
                metrics_to_highlight=request.metrics_to_highlight,
                priorities=request.priorities
            )
            
            return result
        except Exception as e:
            import traceback
            logger.error(f"Service error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    except Exception as e:
        import traceback
        logger = logging.getLogger("picklist_comparison")
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error comparing teams: {str(e)}")