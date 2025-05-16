# backend/app/api/manual_search.py

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from app.services.manual_parser_service import extract_manual_text, analyze_game_manual_in_chunks, search_manual

router = APIRouter(tags=["Manual"])

class ManualUploadRequest(BaseModel):
    manual_url: Optional[str] = None
    year: int = 2025
    use_embeddings: bool = True

class ManualSearchRequest(BaseModel):
    query: str
    year: Optional[int] = None
    top_k: int = 5

@router.post("/api/manual/upload")
async def upload_manual(
    manual_file: Optional[UploadFile] = File(None),
    manual_url: Optional[str] = Form(None),
    year: int = Form(2025),
    use_embeddings: bool = Form(True)
):
    """
    Upload a game manual file or provide a URL to the manual.
    The manual will be processed for text extraction and (optionally) for embeddings.
    """
    if not manual_file and not manual_url:
        raise HTTPException(status_code=400, detail="Either manual_file or manual_url must be provided")
    
    text = await extract_manual_text(manual_file, manual_url, year, use_embeddings)
    
    if not text:
        raise HTTPException(status_code=400, detail="Failed to extract text from the manual")
    
    # Analyze the manual to extract game information
    analysis = await analyze_game_manual_in_chunks(text, year)
    
    return {
        "status": "success",
        "text_length": len(text),
        "year": year,
        "embeddings_processed": use_embeddings,
        "game_info": analysis
    }

@router.post("/api/manual/search")
async def search_game_manual(request: ManualSearchRequest):
    """
    Search the game manual for relevant information using embeddings-based retrieval.
    
    Returns the most relevant chunks of text from the manual based on the query.
    """
    results = await search_manual(request.query, request.year, request.top_k)
    
    if not results:
        return {
            "status": "no_results",
            "message": "No matching content found in the manual",
            "results": []
        }
    
    return {
        "status": "success",
        "result_count": len(results),
        "results": results
    }

@router.get("/api/manual/info")
async def get_manual_info():
    """
    Get information about the currently loaded game manual.
    """
    from app.services.global_cache import cache
    
    manual_text = cache.get("manual_text")
    manual_year = cache.get("manual_year")
    manual_embeddings = cache.get("manual_embeddings")
    
    if not manual_text:
        return {
            "status": "not_found",
            "message": "No manual has been uploaded yet"
        }
    
    return {
        "status": "success",
        "has_text": bool(manual_text),
        "text_length": len(manual_text) if manual_text else 0,
        "year": manual_year,
        "has_embeddings": bool(manual_embeddings),
        "embeddings_info": manual_embeddings
    }