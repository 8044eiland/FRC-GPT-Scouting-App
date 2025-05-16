# backend/app/services/embedding_service.py

import os
import json
import time
import fitz  # PyMuPDF
import numpy as np
import faiss
import asyncio
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union, BinaryIO
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
import hashlib
import logging
from app.services.global_cache import cache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_service.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('embedding_service')

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_indices")

# Constants
CHUNK_SIZE = 1000  # Maximum number of tokens per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
EMBEDDING_DIMENSIONS = 1536  # Dimensions of OpenAI embeddings

# Create directories if they don't exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

async def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_content: Raw bytes of the PDF file
        
    Returns:
        str: Extracted text content with page numbers and section metadata
    """
    pdf_document = fitz.open(stream=file_content, filetype="pdf")
    text_with_metadata = ""
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Extract text from this page
        page_text = page.get_text()
        
        # Add page number as metadata
        text_with_metadata += f"[Page {page_num + 1}]\n{page_text}\n\n"
    
    return text_with_metadata

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
        logger.error(f"Error downloading PDF: {e}")
        return None

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks of approximately equal size.
    Each chunk includes metadata about its source location.
    
    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries with chunk text and metadata
    """
    # Split text into paragraphs
    paragraphs = text.split("\n\n")
    
    chunks = []
    current_chunk = ""
    current_page = 1
    
    for paragraph in paragraphs:
        # Check for page markers
        if paragraph.startswith("[Page "):
            try:
                current_page = int(paragraph.split("]")[0][6:])
                # Skip the page marker paragraph itself
                continue
            except (ValueError, IndexError):
                pass
        
        # If adding this paragraph exceeds the chunk size and we already have content,
        # save the current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": {
                    "page": current_page,
                }
            })
            
            # Start new chunk with overlap from the end of the previous chunk
            # Only if the current chunk is long enough to have meaningful overlap
            if len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "metadata": {
                "page": current_page,
            }
        })
    
    return chunks

async def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of text chunks using OpenAI's API.
    
    Args:
        chunks: List of dictionaries with text chunks and metadata
        
    Returns:
        List of dictionaries with original chunks and their embeddings
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    
    # Create a batch of embedding requests to reduce API calls
    texts = [chunk["text"] for chunk in chunks]
    
    # Process in batches to avoid rate limits
    BATCH_SIZE = 100  # Adjust based on API limits
    embedded_chunks = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_chunks = chunks[i:i+BATCH_SIZE]
        
        try:
            # Call OpenAI API to get embeddings
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts
            )
            
            # Add embeddings to chunks
            for j, chunk in enumerate(batch_chunks):
                try:
                    embedding = response.data[j].embedding
                    
                    # Create a copy of the chunk with its embedding
                    chunk_with_embedding = {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "embedding": embedding
                    }
                    embedded_chunks.append(chunk_with_embedding)
                except IndexError:
                    logger.error(f"Index error for chunk {j} in batch {i//BATCH_SIZE}")
                    continue
            
            # Avoid rate limiting
            if i + BATCH_SIZE < len(texts):
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error generating embeddings for batch starting at {i}: {e}")
            # Continue with the next batch instead of failing completely
            await asyncio.sleep(5)  # Longer cooldown after an error
    
    logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
    return embedded_chunks

def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for fast similarity search with embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        FAISS index with the embeddings added
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Initialize a FAISS index - we use IndexFlatIP for inner product (cosine similarity)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    
    # Normalize the vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Add the vectors to the index
    index.add(embeddings_array)
    
    return index

def save_faiss_index(index: faiss.Index, manual_id: str, year: int) -> str:
    """
    Save a FAISS index to disk.
    
    Args:
        index: FAISS index to save
        manual_id: Unique ID for the manual
        year: FRC season year
        
    Returns:
        Path to the saved index
    """
    index_path = os.path.join(FAISS_DIR, f"manual_{year}_{manual_id}.index")
    faiss.write_index(index, index_path)
    return index_path

def load_faiss_index(manual_id: str, year: int) -> Optional[faiss.Index]:
    """
    Load a FAISS index from disk.
    
    Args:
        manual_id: Unique ID for the manual
        year: FRC season year
        
    Returns:
        FAISS index or None if not found
    """
    index_path = os.path.join(FAISS_DIR, f"manual_{year}_{manual_id}.index")
    if not os.path.exists(index_path):
        return None
    
    try:
        index = faiss.read_index(index_path)
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None

async def process_manual(manual_url: str, year: int) -> Dict[str, Any]:
    """
    Process a game manual from URL: download, extract text, chunk, and generate embeddings.
    
    Args:
        manual_url: URL to the game manual PDF
        year: FRC season year
        
    Returns:
        Dictionary with processing status and statistics
    """
    # Generate a unique ID for this manual based on URL and year
    manual_id = hashlib.md5(f"{manual_url}_{year}".encode()).hexdigest()
    
    # Check if we've already processed this manual (FAISS index exists)
    index_path = os.path.join(FAISS_DIR, f"manual_{year}_{manual_id}.index")
    metadata_path = os.path.join(EMBEDDINGS_DIR, f"manual_{year}_{manual_id}_metadata.json")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        logger.info(f"Using existing FAISS index for manual: {manual_id}")
        return {
            "status": "success",
            "manual_id": manual_id,
            "year": year,
            "message": "Using existing FAISS index"
        }
    
    # Start timing
    start_time = time.time()
    
    # 1. Download the PDF
    logger.info(f"Downloading manual from URL: {manual_url}")
    pdf_content = await fetch_pdf_from_url(manual_url)
    if not pdf_content:
        return {
            "status": "error",
            "message": "Failed to download PDF"
        }
    
    # 2. Extract text from PDF with metadata
    logger.info("Extracting text from PDF")
    manual_text = await extract_text_from_pdf(pdf_content)
    
    # 3. Chunk the text
    logger.info("Chunking text")
    chunks = chunk_text(manual_text)
    logger.info(f"Created {len(chunks)} chunks")
    
    # 4. Generate embeddings
    logger.info("Generating embeddings")
    embedded_chunks = await generate_embeddings(chunks)
    
    if not embedded_chunks:
        return {
            "status": "error",
            "message": "Failed to generate embeddings"
        }
    
    # 5. Build FAISS index
    logger.info("Building FAISS index")
    embeddings_list = [chunk["embedding"] for chunk in embedded_chunks]
    index = build_faiss_index(embeddings_list)
    
    # 6. Save FAISS index
    logger.info(f"Saving FAISS index to {index_path}")
    save_faiss_index(index, manual_id, year)
    
    # 7. Save metadata (chunks without embeddings)
    logger.info(f"Saving metadata to {metadata_path}")
    
    # Remove embeddings from chunks to avoid duplicating data
    metadata_chunks = []
    for i, chunk in enumerate(embedded_chunks):
        metadata_chunks.append({
            "id": i,  # Add an ID to match with FAISS indices
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "manual_id": manual_id,
            "year": year,
            "url": manual_url,
            "chunks": metadata_chunks,
            "created_at": time.time()
        }, f)
    
    # Calculate processing time
    total_time = time.time() - start_time
    
    # Add to cache for quick access
    cache["manual_embeddings"] = {
        "manual_id": manual_id,
        "year": year,
        "url": manual_url,
        "chunk_count": len(embedded_chunks),
        "faiss_index": True
    }
    
    return {
        "status": "success",
        "manual_id": manual_id,
        "year": year,
        "url": manual_url,
        "chunk_count": len(embedded_chunks),
        "processing_time": total_time,
        "faiss_index": True
    }

async def get_chunks_by_query(query: str, year: int, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks for a given query using FAISS vector similarity search.
    
    Args:
        query: The search query
        year: FRC season year to search in
        top_k: Number of most relevant chunks to return
        
    Returns:
        List of the most relevant chunks with their text and metadata
    """
    # Check if we have embeddings for this year
    manual_info = cache.get("manual_embeddings")
    manual_id = None
    
    if manual_info and manual_info.get("year") == year:
        manual_id = manual_info.get("manual_id")
    else:
        # Look for existing FAISS indices for this year
        index_files = [f for f in os.listdir(FAISS_DIR) if f.startswith(f"manual_{year}_") and f.endswith(".index")]
        if not index_files:
            logger.warning(f"No FAISS index found for year {year}")
            return []
        
        # Use the most recent index file
        index_file = sorted(index_files)[-1]
        # Extract manual_id from filename (format: manual_YEAR_ID.index)
        manual_id = index_file.split("_")[-1].replace(".index", "")
    
    if not manual_id:
        logger.warning("No manual ID found for search")
        return []
    
    # Load FAISS index
    index = load_faiss_index(manual_id, year)
    if not index:
        logger.error(f"Failed to load FAISS index for manual ID: {manual_id}")
        return []
    
    # Load metadata
    metadata_path = os.path.join(EMBEDDINGS_DIR, f"manual_{year}_{manual_id}_metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        chunks = metadata.get("chunks", [])
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return []
    
    if not chunks:
        logger.warning("No chunks found in metadata")
        return []
    
    # Generate embedding for query
    try:
        query_response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_embedding = query_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return []
    
    # Convert query embedding to numpy array and normalize
    query_vector = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    # Search the FAISS index
    distances, indices = index.search(query_vector, top_k)
    
    # Create results with text and metadata
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue  # Skip invalid indices
        
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "similarity": float(distances[0][i])  # Convert to Python float for JSON
        })
    
    return results