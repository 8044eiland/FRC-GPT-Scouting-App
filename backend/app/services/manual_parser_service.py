# backend/app/services/manual_parser_service.py

import os
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Any
import io
import json
from fastapi import UploadFile
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx
import logging
from app.services.global_cache import cache
from app.services.embedding_service import process_manual, get_chunks_by_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manual_parser.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('manual_parser')

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_content: Raw bytes of the PDF file
        
    Returns:
        str: Extracted text content
    """
    pdf_document = fitz.open(stream=file_content, filetype="pdf")
    text = ""
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    
    return text

async def fetch_pdf_from_url(url: str) -> Optional[bytes]:
    """
    Download a PDF file from a URL.
    
    Args:
        url: URL pointing to a PDF file
        
    Returns:
        Optional[bytes]: PDF file content or None if download fails
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.content
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None

async def extract_manual_text(manual_file: Optional[UploadFile] = None, manual_url: Optional[str] = None, year: int = 2025, use_embeddings: bool = True) -> Optional[str]:
    """
    Extract text from either an uploaded PDF file or a PDF URL.
    
    Args:
        manual_file: Optional uploaded PDF file
        manual_url: Optional URL to a PDF file
        year: FRC season year
        use_embeddings: Whether to process for vector embeddings
        
    Returns:
        Optional[str]: Extracted text or None if extraction fails
    """
    if manual_file:
        content = await manual_file.read()
        text = await extract_text_from_pdf(content)
        # If it's a file upload, we need to save it somewhere to get a URL for embeddings
        # For simplicity, we'll skip embeddings processing for file uploads in this version
        use_embeddings = False
    elif manual_url:
        pdf_content = await fetch_pdf_from_url(manual_url)
        if not pdf_content:
            return None
        text = await extract_text_from_pdf(pdf_content)
        
        # Process for embeddings if requested
        if use_embeddings:
            logger.info(f"Processing manual for embeddings: {manual_url}, year={year}")
            try:
                result = await process_manual(manual_url, year)
                if result["status"] == "success":
                    logger.info(f"Successfully processed manual embeddings: {result['chunk_count']} chunks")
                    # Store the manual ID for future reference
                    cache["manual_embeddings"] = {
                        "manual_id": result["manual_id"],
                        "year": year,
                        "url": manual_url
                    }
                else:
                    logger.error(f"Failed to process manual embeddings: {result['message']}")
            except Exception as e:
                logger.error(f"Error processing manual embeddings: {e}")
    else:
        return None
    
    # Save to global cache
    cache["manual_text"] = text
    cache["manual_year"] = year
    
    return text

async def search_manual(query: str, year: Optional[int] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the game manual for relevant sections based on a query.
    
    Args:
        query: The search query
        year: FRC season year (defaults to cached year if available)
        top_k: Number of top results to return
        
    Returns:
        List of relevant chunks with their text and metadata
    """
    # Get year from cache if not provided
    if year is None:
        year = cache.get("manual_year", 2025)
    
    # Log the search query
    logger.info(f"Searching manual for '{query}' (year={year})")
    
    # Perform the search
    results = await get_chunks_by_query(query, year, top_k)
    
    if not results:
        logger.warning(f"No results found for query: '{query}'")
    else:
        logger.info(f"Found {len(results)} results for query: '{query}'")
    
    return results

async def extract_game_relevant_sections(manual_text: str, year: int) -> Dict[str, Any]:
    """
    Extract only the game-relevant sections from the manual text.
    Focuses on gameplay, scoring, and field elements.
    
    Args:
        manual_text: Full text of the game manual
        year: FRC season year
        
    Returns:
        Dict with extracted sections and metadata
    """
    # Define keywords for relevant sections
    game_keywords = [
        "game overview", "game description", "gameplay", "match play",
        "scoring", "points", "auto", "autonomous", "teleop", "driver",
        "endgame", "field elements", "game pieces", "alliance", "ranking"
    ]
    
    # Build a prompt to extract relevant sections
    prompt = f"""
You are analyzing an FRC (FIRST Robotics Competition) game manual for the {year} season.
Extract ONLY the most strategically relevant sections related to:

1. Game overview
2. Scoring mechanisms
3. Field elements
4. Auto, teleop, and endgame phases
5. Ranking point criteria

Focus on information that would be relevant for scouting and alliance selection.
Exclude sections about robot construction rules, safety, event rules, or glossaries.

For each extract, include the section title and content. Keep your total response under 2000 tokens.
Format your response as structured JSON:

{{
  "game_name": "name of the game",
  "sections": [
    {{
      "title": "section title",
      "content": "extracted content"
    }}
  ]
}}
"""
    
    try:
        # Call the OpenAI API
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt + "\n\nManual text:\n" + manual_text[:50000]}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        
        # Parse the response
        extracted_data = json.loads(response.choices[0].message.content)
        
        # Create a condensed version of all relevant sections for context
        relevant_sections = ""
        for section in extracted_data.get("sections", []):
            relevant_sections += f"{section['title']}:\n{section['content']}\n\n"
        
        # Add the relevant sections to the result
        result = {
            "game_name": extracted_data.get("game_name", f"FRC {year} Game"),
            "relevant_sections": relevant_sections
        }
        
        # Save the extracted data to a file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        manual_text_path = os.path.join(data_dir, f"manual_text_{year}.json")
        with open(manual_text_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"Error extracting game sections from manual: {e}")
        return {"game_name": f"FRC {year} Game", "relevant_sections": ""}

async def analyze_game_manual_in_chunks(manual_text: str, year: int) -> Dict[str, Any]:
    """
    Analyze the game manual using GPT, processing in smaller chunks to avoid rate limits.
    
    Args:
        manual_text: Extracted text from the game manual
        year: FRC season year
        
    Returns:
        Dict containing extracted game information
    """
    # Try to extract relevant sections first
    try:
        game_info = await extract_game_relevant_sections(manual_text, year)
        if game_info and "game_name" in game_info:
            # We already have the game info from the relevant sections
            return {
                "game_name": game_info.get("game_name"),
                "scouting_variables": await extract_scouting_variables(game_info.get("relevant_sections", ""), year)
            }
    except Exception as e:
        print(f"Error extracting relevant sections: {e}")
    
    # Fallback to basic overview if section extraction fails
    return await analyze_game_overview(year)

async def analyze_game_overview(year: int) -> Dict[str, Any]:
    """
    Process a small sample or overview of the game without requiring the full manual.
    
    Args:
        year: FRC season year
        
    Returns:
        Dict containing basic game information
    """
    # Create a prompt to get basic game info
    prompt = f"""
You are an expert on FIRST Robotics Competition (FRC) games. For the {year} season, provide a basic overview of:
1. The game name
2. Basic field elements
3. Main scoring actions 
4. Common strategy elements relevant to scouting
5. Recommended scouting variables (in snake_case format)

Output structured JSON only, following this format:
{{
  "game_name": "string",
  "field_elements": ["string", "string"],
  "scoring_actions": ["string", "string"],
  "scouting_variables": {{
    "auto_phase": ["auto_variable_1", "auto_variable_2"],
    "teleop_phase": ["teleop_variable_1", "teleop_variable_2"],
    "endgame": ["endgame_variable_1"],
    "strategy": ["strategy_variable_1"]
  }}
}}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        # Parse the response
        extracted_info = json.loads(response.choices[0].message.content)
        
        # Save to cache for reference
        cache["game_analysis"] = extracted_info
        
        return extracted_info
    except Exception as e:
        print(f"Error analyzing game overview: {e}")
        # Return minimal structure if analysis fails
        return {
            "game_name": f"FRC {year} Game",
            "field_elements": [],
            "scoring_actions": [],
            "scouting_variables": {
                "auto_phase": ["auto_score"],
                "teleop_phase": ["teleop_score"],
                "endgame": ["endgame_score"],
                "strategy": ["comments"]
            }
        }

async def extract_scouting_variables(game_text: str, year: int) -> Dict[str, List[str]]:
    """
    Extract recommended scouting variables from game text.
    
    Args:
        game_text: Text describing the game (ideally from relevant sections)
        year: FRC season year
        
    Returns:
        Dict with category keys and lists of variables as values
    """
    prompt = f"""
Based on this description of the {year} FRC game, suggest optimal scouting variables
that teams should track. Group these variables by category and use snake_case format.

Return ONLY a JSON object with this structure:
{{
  "auto_phase": ["auto_variable_1", "auto_variable_2"],
  "teleop_phase": ["teleop_variable_1", "teleop_variable_2"],
  "endgame": ["endgame_variable_1"],
  "strategy": ["strategy_variable_1"]
}}

Include standard variables like team_number and match_number in each appropriate phase.
Focus only on quantifiable metrics and observations that would be useful for alliance selection.

Game Description:
{game_text}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        # Parse the response
        variables_data = json.loads(response.choices[0].message.content)
        
        # Ensure we have at least empty lists for each category
        default_categories = ["auto_phase", "teleop_phase", "endgame", "strategy"]
        for category in default_categories:
            if category not in variables_data:
                variables_data[category] = []
        
        return variables_data
    except Exception as e:
        print(f"Error extracting scouting variables: {e}")
        # Return basic structure if extraction fails
        return {
            "auto_phase": ["auto_score"],
            "teleop_phase": ["teleop_score"],
            "endgame": ["endgame_score"],
            "strategy": ["comments"]
        }