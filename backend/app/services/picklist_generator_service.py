# backend/app/services/picklist_generator_service.py

import json
import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("picklist_generator.log", encoding='utf-8'),  # Explicitly use UTF-8 encoding
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('picklist_generator')

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PicklistGeneratorService:
    """
    Service for generating ranked picklists using GPT, based on team performance
    metrics and alliance strategy priorities.
    """
    
    def __init__(self, unified_dataset_path: str):
        """
        Initialize the picklist generator with the unified dataset.
        
        Args:
            unified_dataset_path: Path to the unified dataset JSON file
        """
        self.dataset_path = unified_dataset_path
        self.dataset = self._load_dataset()
        self.teams_data = self.dataset.get("teams", {})
        self.year = self.dataset.get("year", 2025)
        self.event_key = self.dataset.get("event_key", f"{self.year}arc")
        
        # Load manual text for game context if available
        self.game_context = self._load_game_context()
        
        # Internal cache to avoid duplicate GPT calls
        self._picklist_cache = {}
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the unified dataset from the JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading unified dataset: {e}")
            return {}
    
    def _load_game_context(self) -> Optional[str]:
        """Load the game manual text for context."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            manual_text_path = os.path.join(data_dir, f"manual_text_{self.year}.json")
            
            if os.path.exists(manual_text_path):
                with open(manual_text_path, "r", encoding="utf-8") as f:
                    manual_data = json.load(f)
                    # Combine game name and relevant sections
                    return f"Game: {manual_data.get('game_name', '')}\n\n{manual_data.get('relevant_sections', '')}"
            return None
        except Exception as e:
            print(f"Error loading game context: {e}")
            return None
    
    def _prepare_team_data_for_gpt(self) -> List[Dict[str, Any]]:
        """
        Prepare a condensed version of team data suitable for the GPT context window.
        Includes key metrics and statistics in a structured format.
        
        Based on chunk size guidance, we need to minimize the size of the team data
        to reduce input token count and avoid context limits.
        
        Returns:
            List of dictionaries with team data
        """
        condensed_teams = []
        
        for team_number, team_data in self.teams_data.items():
            # Skip entries that don't have a valid team number
            try:
                team_number_int = int(team_number)
            except (ValueError, TypeError):
                continue
                
            # Create condensed team info with only essential fields
            team_info = {
                "team_number": team_number_int,
                "nickname": team_data.get("nickname", f"Team {team_number}"),
                "metrics": {},
                # Only include match count if we have scouting data
                "match_count": len(team_data.get("scouting_data", [])) if team_data.get("scouting_data") else 0,
            }
            
            # Calculate average metrics from scouting data
            scouting_metrics = {}
            for match in team_data.get("scouting_data", []):
                for key, value in match.items():
                    if isinstance(value, (int, float)) and key not in ["team_number", "match_number", "qual_number"]:
                        if key not in scouting_metrics:
                            scouting_metrics[key] = []
                        scouting_metrics[key].append(value)
            
            # Calculate averages 
            for metric, values in scouting_metrics.items():
                if values:
                    team_info["metrics"][metric] = sum(values) / len(values)
            
            # Add Statbotics metrics if available
            statbotics_info = team_data.get("statbotics_info", {})
            for key, value in statbotics_info.items():
                if isinstance(value, (int, float)):
                    team_info["metrics"][f"statbotics_{key}"] = value
            
            # Add ranking info if available
            ranking_info = team_data.get("ranking_info", {})
            rank = ranking_info.get("rank")
            if rank is not None:
                team_info["rank"] = rank
                
            # Add record if available
            record = ranking_info.get("record")
            if record:
                team_info["record"] = record
            
            # Add qualitative notes from superscouting - but only the most recent one to save tokens
            superscouting_notes = []
            for entry in team_data.get("superscouting_data", [])[:1]:  # Only use the most recent entry
                if "strategy_notes" in entry and entry["strategy_notes"]:
                    superscouting_notes.append(entry["strategy_notes"])
                if "comments" in entry and entry["comments"]:
                    superscouting_notes.append(entry["comments"])
            
            if superscouting_notes:
                team_info["superscouting_notes"] = superscouting_notes[:1]  # Limit to 1 note for context size
            
            condensed_teams.append(team_info)
        
        return condensed_teams
    
    def _get_team_by_number(self, team_number: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific team's data by team number.
        
        Args:
            team_number: The team number to look up
            
        Returns:
            Dictionary with team data or None if not found
        """
        team_str = str(team_number)
        return self.teams_data.get(team_str)
    
    def _calculate_similarity_score(self, team1_metrics: Dict[str, float], team2_metrics: Dict[str, float]) -> float:
        """
        Calculate a similarity score between two teams based on their metrics.
        Higher score means more similar teams.
        
        Args:
            team1_metrics: Metrics dictionary for first team
            team2_metrics: Metrics dictionary for second team
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Find common metrics
        common_metrics = set(team1_metrics.keys()).intersection(set(team2_metrics.keys()))
        
        if not common_metrics:
            return 0.0
        
        similarity = 0.0
        for metric in common_metrics:
            # Calculate normalized difference
            max_val = max(abs(team1_metrics[metric]), abs(team2_metrics[metric]))
            if max_val > 0:  # Avoid division by zero
                diff = abs(team1_metrics[metric] - team2_metrics[metric]) / max_val
                similarity += 1.0 - min(diff, 1.0)  # Cap difference at 1.0
            else:
                similarity += 1.0  # Both values are zero, perfect match
        
        # Average similarity across all metrics
        return similarity / len(common_metrics) if common_metrics else 0.0
    
    async def generate_picklist(
        self,
        your_team_number: int,
        pick_position: str,
        priorities: List[Dict[str, Any]],
        exclude_teams: Optional[List[int]] = None,
        request_id: Optional[int] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete picklist with all teams ranked at once.
        Uses a one-shot approach to rank all teams in a single API call.
        
        Args:
            your_team_number: Your team's number for alliance compatibility
            pick_position: 'first', 'second', or 'third'
            priorities: List of metric IDs and weights to prioritize
            exclude_teams: Optional list of team numbers to exclude (e.g., already picked)
            
        Returns:
            Dict with generated picklist and explanations
        """
        # Check cache first - use provided cache_key if available, otherwise generate one
        if cache_key is None:
            # Default cache key generation (fallback for backward compatibility)
            cache_key = f"{your_team_number}_{pick_position}_{json.dumps(priorities)}_{json.dumps(exclude_teams or [])}"
        
        # Add request_id to log messages if provided
        request_info = f" [Request: {request_id}]" if request_id is not None else ""
        
        # Add a timestamp to check for active in-progress generations
        current_time = time.time()
        
        if cache_key in self._picklist_cache:
            cached_result = self._picklist_cache[cache_key]
            
            # Check if this is an in-progress generation (indicated by a timestamp value)
            if isinstance(cached_result, float):
                # If the generation started less than 2 minutes ago, wait for it to complete
                if current_time - cached_result < 120:  # 2 minute timeout
                    logger.info(f"Detected in-progress picklist generation for same parameters{request_info}, waiting for completion...")
                    
                    # Wait for a short time to allow the other process to finish
                    for _ in range(12):  # Try for up to 1 minute
                        await asyncio.sleep(5)  # Wait 5 seconds between checks
                        
                        # Check if the result is now available
                        if cache_key in self._picklist_cache and not isinstance(self._picklist_cache[cache_key], float):
                            logger.info(f"Successfully retrieved result from parallel generation{request_info}")
                            return self._picklist_cache[cache_key]
                    
                    # If we get here, the other process took too long or failed
                    logger.warning(f"Timeout waiting for parallel generation, proceeding with new generation{request_info}")
                    # Fall through to generate a new result
                else:
                    # The previous generation is stale, remove it and continue
                    logger.warning(f"Found stale in-progress picklist generation, starting fresh{request_info}")
                    del self._picklist_cache[cache_key]
            else:
                # We have a valid cached result
                logger.info(f"Using cached picklist{request_info}")
                return cached_result
                
        # Mark this cache key as "in progress" by storing the current timestamp
        self._picklist_cache[cache_key] = current_time
        
        # Get your team data
        your_team = self._get_team_by_number(your_team_number)
        if not your_team:
            return {
                "status": "error",
                "message": f"Your team {your_team_number} not found in dataset"
            }
        
        # Prepare team data for GPT
        teams_data = self._prepare_team_data_for_gpt()
        
        # Filter out excluded teams
        if exclude_teams:
            teams_data = [team for team in teams_data if team["team_number"] not in exclude_teams]
        
        try:
            # Start comprehensive logging
            logger.info(f"====== STARTING PICKLIST GENERATION ======")
            logger.info(f"Pick position: {pick_position}")
            logger.info(f"Your team: {your_team_number}")
            logger.info(f"Priority metrics count: {len(priorities)}")
            logger.info(f"Total teams to rank: {len(teams_data)}")
            if exclude_teams:
                logger.info(f"Excluded teams: {exclude_teams}")
            
            # Initialize variables for one-shot approach
            start_time = time.time()
            estimated_time = len(teams_data) * 0.9  # ~0.9 seconds per team estimate
            logger.info(f"Estimated completion time: {estimated_time:.1f} seconds")
            
            # Create prompts optimized for one-shot completion
            system_prompt = self._create_system_prompt(pick_position, len(teams_data))
            
            # Get sorted list of team numbers for verification
            team_numbers = sorted([team["team_number"] for team in teams_data])
            logger.info(f"Teams to rank: {len(team_numbers)}")
            logger.info(f"Team numbers: {team_numbers[:10]}... (and {len(team_numbers) - 10} more)")
            
            user_prompt = self._create_user_prompt(
                your_team_number, 
                pick_position, 
                priorities, 
                teams_data,
                team_numbers
            )
            
            # Log prompts (truncated for readability but showing structure)
            truncated_system = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt
            logger.info(f"SYSTEM PROMPT (truncated):\n{truncated_system}")
            
            # Log just the structure of the user prompt, not the full team data which would be too large
            user_prompt_structure = "\n".join(user_prompt.split("\n")[:10]) + "\n...[Team data truncated]..."
            logger.info(f"USER PROMPT (structure):\n{user_prompt_structure}")
            
            # Initialize messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make a single API call to rank all teams at once
            logger.info(f"--- Requesting complete picklist for {len(team_numbers)} teams ---")
            request_start_time = time.time()
            
            # Call GPT with optimized settings for one-shot generation
            logger.info("Starting API call...")
            
            # Use exponential backoff for rate limit handling
            max_retries = 3
            initial_delay = 1.0  # Start with a 1-second delay
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",  # Using a model that can handle the full team list
                        messages=messages,
                        temperature=0.2,  # Lower temperature for more consistent results
                        response_format={"type": "json_object"},
                        max_tokens=4000  # Increased to prevent truncation with the compact schema
                    )
                    # Success - break out of the retry loop
                    break
                except Exception as e:
                    # Check if it's a rate limit error (typically a 429 status code)
                    is_rate_limit = "429" in str(e)
                    
                    if is_rate_limit and retry_count < max_retries:
                        # Calculate exponential backoff delay
                        retry_count += 1
                        delay = initial_delay * (2 ** retry_count)  # Exponential backoff
                        
                        logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                    else:
                        # Either not a rate limit error or we've exceeded max retries
                        logger.error(f"API call failed: {str(e)}")
                        raise  # Re-raise the exception
            
            # Log timing and response metadata
            request_time = time.time() - request_start_time
            logger.info(f"Total response time: {request_time:.2f}s (avg: {request_time/len(team_numbers):.2f}s per team)")
            logger.info(f"Response metadata: finish_reason={response.choices[0].finish_reason}, model={response.model}")
            
            # Parse the response
            response_content = response.choices[0].message.content
            
            # Log a sample of the response (first 200 chars)
            response_sample = response_content[:200] + "..." if len(response_content) > 200 else response_content
            logger.info(f"Response sample: {response_sample}")
            
            # Parse the JSON response
            try:
                # Log the full raw response (limited to 1000 chars) for debugging purposes
                if len(response_content) > 1000:
                    logger.info(f"Raw JSON response (first 1000 chars):\n{response_content[:1000]}")
                else:
                    logger.info(f"Raw JSON response:\n{response_content}")
                
                response_data = json.loads(response_content)
                
                # Check for overflow condition in ultra-compact format
                if response_data.get("s") == "overflow":
                    logger.warning("GPT returned overflow status - token limit reached")
                    return {
                        "status": "error",
                        "message": "The amount of team data exceeded the token limit. Please try with fewer teams or simplified priorities."
                    }
                # Check for overflow in regular format
                elif response_data.get("status") == "overflow":
                    logger.warning("GPT returned overflow status - token limit reached")
                    return {
                        "status": "error",
                        "message": "The amount of team data exceeded the token limit. Please try with fewer teams or simplified priorities."
                    }
                
                # Handle ultra-compact format {"p":[[team,score,"reason"]...],"s":"ok"}
                if "p" in response_data and isinstance(response_data["p"], list):
                    logger.info(f"Response contains {len(response_data['p'])} teams in ultra-compact format")
                    
                    # Log first few teams for debugging
                    teams_sample = response_data["p"][:3]
                    logger.info(f"First few teams (ultra-compact): {json.dumps(teams_sample)}")
                    
                    # Check for repeating patterns in teams
                    team_nums = [int(entry[0]) for entry in response_data["p"] if len(entry) >= 1]
                    team_counts = {}
                    for team_num in team_nums:
                        team_counts[team_num] = team_counts.get(team_num, 0) + 1
                    
                    # Check if we have duplicates
                    duplicates = {team: count for team, count in team_counts.items() if count > 1}
                    if duplicates:
                        logger.warning(f"Response contains duplicates: {duplicates}")
                        logger.warning(f"First 20 team numbers: {team_nums[:20]}")
                        
                        # Check if we have a repeating pattern
                        if len(team_nums) > 16:
                            # Check for common sequence lengths
                            for pattern_length in [4, 8, 12, 16]:
                                if len(team_nums) >= pattern_length * 2:
                                    pattern1 = team_nums[:pattern_length]
                                    pattern2 = team_nums[pattern_length:pattern_length*2]
                                    if pattern1 == pattern2:
                                        logger.warning(f"Detected repeating pattern of length {pattern_length}")
                                        logger.warning(f"Model is repeating teams instead of ranking all teams")
                                        # Truncate to first pattern only to avoid duplicates
                                        logger.warning(f"Truncating response to first {pattern_length} teams")
                                        response_data["p"] = response_data["p"][:pattern_length]
                                        break
                    
                    # Calculate duplication percentage
                    total_entries = len(team_nums)
                    unique_teams = len(team_counts)
                    if total_entries > 0:  # Prevent division by zero
                        duplication_percentage = ((total_entries - unique_teams) / total_entries) * 100
                        logger.warning(f"Duplication percentage: {duplication_percentage:.1f}%")
                        
                        # If duplication is extreme (over 80%), warn that model might be in a loop
                        if duplication_percentage > 80 and total_entries > 30:
                            logger.error(f"MODEL APPEARS TO BE LOOPING - {duplication_percentage:.1f}% duplicates")
                            return {
                                "status": "error",
                                "message": "The model is unable to rank all teams at once. Please try reducing the number of teams to rank (e.g., 25 at a time) or simplify the priorities. The model got stuck repeating the same teams."
                            }
                    
                    # Convert ultra-compact format to standard format
                    picklist = []
                    seen_teams = set()  # Track teams we've already added
                    
                    for team_entry in response_data["p"]:
                        if len(team_entry) >= 3:  # Ensure we have at least [team, score, reason]
                            team_number = int(team_entry[0])
                            
                            # Skip if we've seen this team already
                            if team_number in seen_teams:
                                logger.info(f"Skipping duplicate team {team_number} in response")
                                continue
                                
                            seen_teams.add(team_number)
                            score = float(team_entry[1])
                            reason = team_entry[2]
                            
                            # Get team nickname from dataset if available
                            team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                            nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                            
                            picklist.append({
                                "team_number": team_number,
                                "nickname": nickname,
                                "score": score,
                                "reasoning": reason
                            })
                            
                # Handle regular compact format
                elif "picklist" in response_data and isinstance(response_data["picklist"], list):
                    logger.info(f"Response contains {len(response_data['picklist'])} teams in regular format")
                    
                    # Log first few teams for debugging
                    teams_sample = response_data["picklist"][:3]
                    logger.info(f"First few teams: {json.dumps(teams_sample)}")
                    
                    # Extract the picklist and convert from compact to full format if needed
                    raw_picklist = response_data.get("picklist", [])
                    
                    # Convert compact format {"team":123, "score":45.6, "reason":"text"} 
                    # to standard format {"team_number":123, "nickname":"Team 123", "score":45.6, "reasoning":"text"}
                    picklist = []
                    for team_entry in raw_picklist:
                        # Check if using new compact format (has "team" instead of "team_number")
                        if "team" in team_entry and "team_number" not in team_entry:
                            team_number = team_entry["team"]
                            # Get team nickname from dataset if available
                            team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                            nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                            
                            picklist.append({
                                "team_number": team_number,
                                "nickname": nickname,
                                "score": team_entry.get("score", 0.0),
                                "reasoning": team_entry.get("reason", "No reasoning provided")
                            })
                        else:
                            # Already in standard format
                            picklist.append(team_entry)
                else:
                    logger.warning("Response has no valid picklist")
                    picklist = []
                
                # Create minimal analysis since we removed it from the schema
                analysis = {
                    "draft_reasoning": "Analysis not included to optimize token usage",
                    "evaluation": "Analysis not included to optimize token usage",
                    "final_recommendations": "Analysis not included to optimize token usage"
                }
                
            except json.JSONDecodeError as e:
                # Log the error and the full response for debugging
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Full response that couldn't be parsed: {response_content}")
                
                # Try to fix common JSON issues and salvage what we can
                try:
                    # Try to use a more lenient JSON parser
                    import re
                    import ast
                    
                    # Simple fix for unescaped quotes
                    fixed_content = re.sub(r'(?<!\\)"(?=(,|\]|\}|:))', r'\\"', response_content)
                    
                    # Try parsing again
                    logger.info("Attempting to repair the JSON response...")
                    response_data = json.loads(fixed_content)
                    
                    # If we get here, the fix worked
                    logger.info("Successfully repaired JSON response!")
                    picklist = response_data.get("picklist", [])
                    analysis = response_data.get("analysis", {
                        "draft_reasoning": "Analysis not available",
                        "evaluation": "Analysis not available",
                        "final_recommendations": "Analysis not available"
                    })
                    
                except Exception as repair_error:
                    # If repair fails, construct a fallback response
                    logger.error(f"JSON repair failed: {repair_error}")
                    
                    # Extract any team data we can using regex
                    try:
                        logger.info("Attempting to extract team data from broken JSON...")
                        
                        # Try to extract from ultra-compact format first
                        # Format: [teamnum,score,"reason"] in a p array
                        compact_pattern = r'\[\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*"([^"]*)"\s*\]'
                        compact_teams_extracted = re.findall(compact_pattern, response_content)
                        
                        if compact_teams_extracted:
                            logger.info(f"Extracted {len(compact_teams_extracted)} team entries from broken ultra-compact JSON")
                            
                            # Log the first few raw extractions for debugging
                            for i, team_raw in enumerate(compact_teams_extracted[:3]):
                                logger.info(f"Raw extraction {i+1} (ultra-compact): {team_raw}")
                                
                            picklist = []
                            team_numbers_seen = set()  # Track team numbers to detect duplicates in regex extraction
                            
                            for team_match in compact_teams_extracted:
                                try:
                                    team_number = int(team_match[0])
                                    score = float(team_match[1])
                                    reasoning = team_match[2]
                                    
                                    # Skip obvious duplicates during extraction
                                    if team_number in team_numbers_seen:
                                        logger.info(f"Skipping duplicate team {team_number} during regex extraction")
                                        continue
                                    
                                    team_numbers_seen.add(team_number)
                                    
                                    # Get team nickname from dataset if available
                                    team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                                    nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                                    
                                    picklist.append({
                                        "team_number": team_number,
                                        "nickname": nickname,
                                        "score": score,
                                        "reasoning": reasoning
                                    })
                                except Exception as team_error:
                                    logger.error(f"Error parsing team data: {team_error}")
                                    continue
                                    
                            analysis = {
                                "draft_reasoning": "Analysis not available - recovered from parsing error",
                                "evaluation": "Analysis not available - recovered from parsing error",
                                "final_recommendations": "Analysis not available - recovered from parsing error"
                            }
                            
                            logger.info(f"Salvaged {len(picklist)} teams from broken response")
                        else:
                            # If not found, try the regular compact format
                            team_pattern1 = r'"team":\s*(\d+),\s*"score":\s*([\d\.]+),\s*"reason":\s*"([^"]*)"'
                            team_pattern2 = r'"team_number":\s*(\d+),\s*"nickname":\s*"([^"]*)",\s*"score":\s*([\d\.]+),\s*"reasoning":\s*"([^"]*)"'
                            
                            teams_extracted1 = re.findall(team_pattern1, response_content)
                            teams_extracted2 = re.findall(team_pattern2, response_content)
                            
                            if teams_extracted1:
                                logger.info(f"Extracted {len(teams_extracted1)} team entries from broken compact JSON")
                                
                                # Log the first few raw extractions for debugging
                                for i, team_raw in enumerate(teams_extracted1[:3]):
                                    logger.info(f"Raw extraction {i+1} (compact): {team_raw}")
                                    
                                picklist = []
                                team_numbers_seen = set()  # Track team numbers to detect duplicates 
                                
                                for team_match in teams_extracted1:
                                    try:
                                        team_number = int(team_match[0])
                                        score = float(team_match[1])
                                        reasoning = team_match[2]
                                        
                                        # Skip obvious duplicates during extraction
                                        if team_number in team_numbers_seen:
                                            logger.info(f"Skipping duplicate team {team_number} during regex extraction")
                                            continue
                                        
                                        team_numbers_seen.add(team_number)
                                        
                                        # Get team nickname from dataset if available
                                        team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                                        nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                                        
                                        picklist.append({
                                            "team_number": team_number,
                                            "nickname": nickname,
                                            "score": score,
                                            "reasoning": reasoning
                                        })
                                    except Exception as team_error:
                                        logger.error(f"Error parsing team data: {team_error}")
                                        continue
                                
                                analysis = {
                                    "draft_reasoning": "Analysis not available - recovered from parsing error",
                                    "evaluation": "Analysis not available - recovered from parsing error",
                                    "final_recommendations": "Analysis not available - recovered from parsing error"
                                }
                                
                                logger.info(f"Salvaged {len(picklist)} teams from broken response")
                            elif teams_extracted2:
                                logger.info(f"Extracted {len(teams_extracted2)} team entries from broken standard JSON")
                                
                                # Log the first few raw extractions for debugging
                                for i, team_raw in enumerate(teams_extracted2[:3]):
                                    logger.info(f"Raw extraction {i+1}: {team_raw}")
                                    
                                picklist = []
                                team_numbers_seen = set()  # Track team numbers to detect duplicates 
                                
                                for team_match in teams_extracted2:
                                    try:
                                        team_number = int(team_match[0])
                                        team_name = team_match[1]
                                        score = float(team_match[2])
                                        reasoning = team_match[3]
                                        
                                        # Skip obvious duplicates during extraction
                                        if team_number in team_numbers_seen:
                                            logger.info(f"Skipping duplicate team {team_number} during regex extraction")
                                            continue
                                        
                                        team_numbers_seen.add(team_number)
                                        
                                        picklist.append({
                                            "team_number": team_number,
                                            "nickname": team_name,
                                            "score": score,
                                            "reasoning": reasoning
                                        })
                                    except Exception as team_error:
                                        logger.error(f"Error parsing team data: {team_error}")
                                        continue
                                
                                analysis = {
                                    "draft_reasoning": "Analysis not available - recovered from parsing error",
                                    "evaluation": "Analysis not available - recovered from parsing error",
                                    "final_recommendations": "Analysis not available - recovered from parsing error"
                                }
                                
                                logger.info(f"Salvaged {len(picklist)} teams from broken response")
                            else:
                                # If we couldn't extract any teams, re-raise the original error
                                logger.error("Could not extract any team data from the broken response")
                                raise e
                    except Exception as extract_error:
                        logger.error(f"Failed to extract team data: {extract_error}")
                        raise e
            
            # Process the picklist
            logger.info("=== Processing picklist results ===")
            logger.info(f"Total teams received: {len(picklist)}")
            
            # Check for duplicate teams and handle them intelligently
            team_entries = {}  # Map team numbers to their entries
            duplicates = []
            
            for team in picklist:
                team_number = team.get("team_number")
                if not team_number:
                    continue  # Skip teams without a valid team number
                
                if team_number not in team_entries:
                    # First time seeing this team
                    team_entries[team_number] = team
                else:
                    # Found a duplicate team - keep the one with the higher score
                    duplicates.append(team_number)
                    current_score = team_entries[team_number].get("score", 0)
                    new_score = team.get("score", 0)
                    
                    if new_score > current_score:
                        # This new entry has a higher score, use it instead
                        logger.info(f"Team {team_number} appears twice - keeping entry with higher score ({new_score} vs {current_score})")
                        team_entries[team_number] = team
            
            # Create the deduplicated picklist from the team_entries map
            deduplicated_picklist = list(team_entries.values())
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} duplicate teams: {duplicates[:10]}...")
                logger.info(f"Resolved by keeping the entry with higher score for each team")
                
                # Analyze the duplicates in more detail
                duplicate_counts = {}
                for team_num in duplicates:
                    if team_num not in duplicate_counts:
                        duplicate_counts[team_num] = 0
                    duplicate_counts[team_num] += 1
                
                # Find teams with the most duplicates
                sorted_duplicates = sorted(duplicate_counts.items(), key=lambda x: x[1], reverse=True)
                if sorted_duplicates:
                    logger.info(f"Most duplicated teams: {sorted_duplicates[:5]}")
                    
                    # Log positions of a highly duplicated team
                    if sorted_duplicates[0][1] > 1:
                        most_duplicated = sorted_duplicates[0][0]
                        positions = [i for i, team in enumerate(picklist) if team.get('team_number') == most_duplicated]
                        logger.info(f"Team {most_duplicated} appears at positions: {positions}")
            
            logger.info(f"After deduplication: {len(deduplicated_picklist)} teams")
            
            # Check if we're missing teams
            available_team_numbers = {team["team_number"] for team in teams_data}
            ranked_team_numbers = {team["team_number"] for team in deduplicated_picklist}
            missing_team_numbers = available_team_numbers - ranked_team_numbers
            
            # Debug: Log the first 5 teams in the picklist before deduplication
            if picklist and len(picklist) > 0:
                logger.info(f"First 5 raw teams from GPT response BEFORE deduplication:")
                for i, team in enumerate(picklist[:5]):
                    logger.info(f"  Raw Team {i+1}: {team.get('team_number')} - {team.get('nickname')}")
                
                # Also log team numbers in sequence to check for patterns in the duplicates
                team_numbers_sequence = [t.get('team_number') for t in picklist[:20]]
                logger.info(f"First 20 team numbers in response: {team_numbers_sequence}")
            
            # Log the completeness
            coverage_percent = (len(ranked_team_numbers) / len(available_team_numbers)) * 100 if available_team_numbers else 0
            logger.info(f"GPT coverage: {coverage_percent:.1f}% ({len(ranked_team_numbers)} of {len(available_team_numbers)} teams)")
            
            # If we're missing teams, add them to the end
            if missing_team_numbers:
                logger.warning(f"Missing {len(missing_team_numbers)} teams from GPT response")
                if len(missing_team_numbers) <= 10:  # Only log all missing teams if there aren't too many
                    logger.warning(f"Missing team numbers: {sorted(list(missing_team_numbers))}")
                else:
                    logger.warning(f"First 10 missing team numbers: {sorted(list(missing_team_numbers))[:10]}...")
                
                # Get the lowest score from the existing picklist
                min_score = min([team["score"] for team in deduplicated_picklist]) if deduplicated_picklist else 0.0
                backup_score = max(0.1, min_score * 0.5)  # Use half of the minimum score or 0.1, whichever is higher
                logger.info(f"Using backup score {backup_score} for missing teams")
                
                # Add missing teams to the end of the picklist
                for team_number in missing_team_numbers:
                    # Find the team data
                    team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                    if team_data:
                        # Add to the end with a lower score
                        deduplicated_picklist.append({
                            "team_number": team_number,
                            "nickname": team_data.get("nickname", f"Team {team_number}"),
                            "score": backup_score,
                            "reasoning": "Added to complete the picklist - not enough data available for detailed analysis",
                            "is_fallback": True  # Flag to indicate this team was added as a fallback
                        })
            
            # Assemble final result
            result = {
                "status": "success",
                "picklist": deduplicated_picklist,
                "analysis": analysis,
                "missing_team_numbers": list(missing_team_numbers) if missing_team_numbers else [],
                "performance": {
                    "total_time": request_time,
                    "team_count": len(team_numbers),
                    "avg_time_per_team": request_time / len(team_numbers) if team_numbers else 0,
                    "missing_teams": len(missing_team_numbers),
                    "duplicate_teams": len(duplicates)
                }
            }
            
            # Log completion stats
            total_time = time.time() - start_time
            logger.info(f"====== PICKLIST GENERATION COMPLETE ======")
            logger.info(f"Total time: {total_time:.2f}s for {len(deduplicated_picklist)} teams")
            logger.info(f"Average time per team: {(total_time / len(deduplicated_picklist) if deduplicated_picklist else 0):.2f}s")
            logger.info(f"Final picklist length: {len(deduplicated_picklist)}")
            
            # Cache the result, replacing the "in progress" timestamp
            self._picklist_cache[cache_key] = result
            
            logger.info(f"Successfully completed picklist generation{request_info}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating picklist with GPT: {str(e)}{request_info}", exc_info=True)
            
            # Clean up the in-progress flag from cache so future requests can proceed
            if cache_key in self._picklist_cache and isinstance(self._picklist_cache[cache_key], float):
                del self._picklist_cache[cache_key]
                
            return {
                "status": "error",
                "message": f"Failed to generate picklist: {str(e)}"
            }
    
    def _create_system_prompt(self, pick_position: str, team_count: int) -> str:
        """
        Create the system prompt for GPT based on the pick position.
        Optimized for one-shot generation of the complete picklist.
        
        Args:
            pick_position: 'first', 'second', or 'third'
            team_count: Total number of teams to rank
            
        Returns:
            System prompt string
        """
        position_context = {
            "first": "First pick teams should be overall powerhouse teams that excel in multiple areas.",
            "second": "Second pick teams should complement the first pick and address specific needs.",
            "third": "Third pick teams are more specialized, often focusing on a single critical function."
        }
        
        return f"""You are GPT-4o, an FRC pick-list strategist.

TASK
Rank ALL {team_count} unique teams for the {pick_position} pick of Team {{your_team_number}}.
Return MINIFIED JSON, ONE LINE, NO SPACES/NEWLINES using THIS exact shape:

{{"p":[[team,score,"≤12 words"]...],"s":"ok"}}

⚠️ CRITICAL RULES:
1. Each team MUST appear EXACTLY ONCE. NO DUPLICATES ALLOWED!
2. Include ALL {team_count} teams from TEAM_NUMBERS_TO_INCLUDE.
3. Each reason must be ≤ 12 words and cite ≥ 1 metric value.
4. NO whitespaces, tabs, or line-breaks in the output.
5. If you cannot fit ALL {team_count} teams, STOP and ONLY return: {{"s":"overflow"}}

ULTRA-COMPACT STRUCTURE EXPLANATION:
- "p" is the array of team entries (replaces "picklist")
- Each team is [team_number, score, "reason"] (array instead of object)
- "s" is status ("ok" or "overflow")

TOKEN OPTIMIZATION EXAMPLES (use different teams):
- [254, 98.3, "Strong autonomous EPA 25.7"]
- [1678, 92.1, "Excellent climb consistency 96%"]
- [3310, 87.5, "High teleop scoring average 15.2 points"]
- [118, 85.2, "Great defense rating 8.9"]

VALIDATION REQUIREMENTS:
1. Verify ALL {team_count} teams are included - CHECK CLOSELY!
2. Check for duplicated team numbers - NONE ALLOWED!
3. Verify your reasoning strings are ≤ 12 words each
4. Ensure JSON is complete, valid, and has NO WHITESPACE

Additional context: {position_context.get(pick_position, "")}

OVERFLOW HANDLING: If you cannot include all {team_count} teams, return ONLY {{"s":"overflow"}} - nothing else.
"""
    
    def _create_user_prompt(
        self, 
        your_team_number: int, 
        pick_position: str, 
        priorities: List[Dict[str, Any]],
        teams_data: List[Dict[str, Any]],
        team_numbers: List[int] = None
    ) -> str:
        """
        Create the user prompt for GPT with all necessary context.
        
        Args:
            your_team_number: Your team's number
            pick_position: 'first', 'second', or 'third'
            priorities: List of metric priorities with weights
            teams_data: Prepared team data for context
            team_numbers: List of team numbers to verify inclusion
            
        Returns:
            User prompt string
        """
        # Find your team's data
        your_team_info = next((team for team in teams_data if team["team_number"] == your_team_number), None)
        
        prompt = f"""YOUR_TEAM_PROFILE = {json.dumps(your_team_info, indent=2) if your_team_info else "{}"} 
PRIORITY_METRICS  = {json.dumps(priorities, indent=2)}
GAME_CONTEXT      = {json.dumps(self.game_context) if self.game_context else "null"}
TEAM_NUMBERS_LIST = {json.dumps(team_numbers)}

Instructions:
1. You MUST rank ALL {len(teams_data)} unique teams listed in TEAM_NUMBERS_LIST.
2. Each team MUST appear EXACTLY ONCE in your output. DO NOT REPEAT TEAMS!
3. Use metrics from AVAILABLE_TEAMS to evaluate each team's performance.
4. Assess alliance synergy with Team {your_team_number}.
5. Sort teams from best→worst based on weighted metrics.
6. Verify your output contains EACH team from TEAM_NUMBERS_LIST EXACTLY ONCE.
7. Return a single JSON object with the "p" array containing ALL {len(teams_data)} teams.

⚠️ CRITICAL REQUIREMENTS:
- NO DUPLICATES: Verify each team number appears only once
- COMPLETENESS: Include ALL {len(teams_data)} teams from TEAM_NUMBERS_LIST
- FORMAT: Follow the ultra-compact schema from the system prompt
- BREVITY: Keep reasons ≤ 12 words each

AVAILABLE_TEAMS   = {json.dumps(teams_data, indent=2)}

End of prompt.
"""
        return prompt

    async def rank_missing_teams(
        self,
        missing_team_numbers: List[int],
        ranked_teams: List[Dict[str, Any]],
        your_team_number: int,
        pick_position: str,
        priorities: List[Dict[str, Any]],
        batch_size: int = 20,  # Smaller batch size to avoid rate limits and token limits
        request_id: Optional[int] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate rankings for teams that were missed in the initial picklist generation.
        
        Args:
            missing_team_numbers: List of team numbers that need to be ranked
            ranked_teams: List of teams already ranked (provides context)
            your_team_number: Your team's number for alliance compatibility
            pick_position: 'first', 'second', or 'third'
            priorities: List of metric IDs and weights to prioritize
            
        Returns:
            Dict with rankings for the previously missing teams
        """
        # Check cache first - use provided cache_key if available, otherwise generate one
        if cache_key is None:
            # Default cache key generation (fallback for backward compatibility)
            cache_key = f"missing_{your_team_number}_{pick_position}_{json.dumps(priorities)}_{json.dumps(missing_team_numbers)}"
        
        # Add request_id to log messages if provided
        request_info = f" [Request: {request_id}]" if request_id is not None else ""
        
        # Add a timestamp to check for active in-progress generations
        current_time = time.time()
        
        if cache_key in self._picklist_cache:
            cached_result = self._picklist_cache[cache_key]
            
            # Check if this is an in-progress generation (indicated by a timestamp value)
            if isinstance(cached_result, float):
                # If the generation started less than 2 minutes ago, wait for it to complete
                if current_time - cached_result < 120:  # 2 minute timeout
                    logger.info(f"Detected in-progress missing teams ranking for same parameters{request_info}, waiting for completion...")
                    
                    # Wait for a short time to allow the other process to finish
                    for _ in range(12):  # Try for up to 1 minute
                        await asyncio.sleep(5)  # Wait 5 seconds between checks
                        
                        # Check if the result is now available
                        if cache_key in self._picklist_cache and not isinstance(self._picklist_cache[cache_key], float):
                            logger.info(f"Successfully retrieved result from parallel missing teams ranking{request_info}")
                            return self._picklist_cache[cache_key]
                    
                    # If we get here, the other process took too long or failed
                    logger.warning(f"Timeout waiting for parallel missing teams ranking, proceeding with new generation{request_info}")
                    # Fall through to generate a new result
                else:
                    # The previous generation is stale, remove it and continue
                    logger.warning(f"Found stale in-progress missing teams ranking, starting fresh{request_info}")
                    del self._picklist_cache[cache_key]
            else:
                # We have a valid cached result
                logger.info(f"Using cached missing teams ranking{request_info}")
                return cached_result
                
        # Mark this cache key as "in progress" by storing the current timestamp
        self._picklist_cache[cache_key] = current_time
        
        # Get your team data
        your_team = self._get_team_by_number(your_team_number)
        if not your_team:
            return {
                "status": "error",
                "message": f"Your team {your_team_number} not found in dataset"
            }
        
        # Prepare team data for GPT (only for missing teams)
        all_teams_data = self._prepare_team_data_for_gpt()
        teams_data = [team for team in all_teams_data if team["team_number"] in missing_team_numbers]
        
        if not teams_data:
            return {
                "status": "error",
                "message": f"No team data found for missing teams"
            }
        
        try:
            # Start comprehensive logging
            logger.info(f"====== STARTING MISSING TEAMS RANKING ======")
            logger.info(f"Pick position: {pick_position}")
            logger.info(f"Your team: {your_team_number}")
            logger.info(f"Priority metrics count: {len(priorities)}")
            logger.info(f"Missing teams to rank: {len(teams_data)}")
            logger.info(f"Missing team numbers: {missing_team_numbers}")
            
            # Initialize variables
            start_time = time.time()
            estimated_time = len(teams_data) * 0.5  # ~0.5 seconds per team (faster for smaller batch)
            logger.info(f"Estimated completion time: {estimated_time:.1f} seconds")
            
            # Create specialized system prompt for missing teams
            system_prompt = self._create_missing_teams_system_prompt(pick_position, len(teams_data))
            
            # Create specialized user prompt for missing teams that includes already-ranked teams
            user_prompt = self._create_missing_teams_user_prompt(
                your_team_number,
                pick_position,
                priorities,
                teams_data,
                ranked_teams
            )
            
            # Log prompts (truncated for readability)
            truncated_system = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt
            logger.info(f"MISSING TEAMS SYSTEM PROMPT (truncated):\n{truncated_system}")
            
            # Initialize messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make a single API call to rank missing teams
            logger.info(f"--- Requesting rankings for {len(missing_team_numbers)} missing teams ---")
            request_start_time = time.time()
            
            # Call GPT with optimized settings
            logger.info("Starting API call...")
            
            # Use exponential backoff for rate limit handling
            max_retries = 3
            initial_delay = 1.0  # Start with a 1-second delay
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                        max_tokens=4000  # Increased to match main method
                    )
                    # Success - break out of the retry loop
                    break
                except Exception as e:
                    # Check if it's a rate limit error (typically a 429 status code)
                    is_rate_limit = "429" in str(e)
                    
                    if is_rate_limit and retry_count < max_retries:
                        # Calculate exponential backoff delay
                        retry_count += 1
                        delay = initial_delay * (2 ** retry_count)  # Exponential backoff
                        
                        logger.warning(f"Rate limit hit when ranking missing teams. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                    else:
                        # Either not a rate limit error or we've exceeded max retries
                        logger.error(f"API call for missing teams failed: {str(e)}")
                        raise  # Re-raise the exception
            
            # Log timing and response metadata
            request_time = time.time() - request_start_time
            logger.info(f"Total response time: {request_time:.2f}s (avg: {request_time/len(teams_data):.2f}s per team)")
            logger.info(f"Response metadata: finish_reason={response.choices[0].finish_reason}, model={response.model}")
            
            # Parse the response
            response_content = response.choices[0].message.content
            response_sample = response_content[:200] + "..." if len(response_content) > 200 else response_content
            logger.info(f"Response sample: {response_sample}")
            
            # Parse the JSON response with error handling
            try:
                # Log the full raw response (limited to 1000 chars) for debugging purposes
                if len(response_content) > 1000:
                    logger.info(f"Raw JSON response (first 1000 chars):\n{response_content[:1000]}")
                else:
                    logger.info(f"Raw JSON response:\n{response_content}")
                
                response_data = json.loads(response_content)
                
                # Check for overflow condition in ultra-compact format
                if response_data.get("s") == "overflow":
                    logger.warning("GPT returned overflow status - token limit reached")
                    return {
                        "status": "error",
                        "message": "The amount of team data exceeded the token limit. Please try with fewer teams or simplified priorities."
                    }
                # Check for overflow in regular format
                elif response_data.get("status") == "overflow":
                    logger.warning("GPT returned overflow status - token limit reached")
                    return {
                        "status": "error",
                        "message": "The amount of team data exceeded the token limit. Please try with fewer teams or simplified priorities."
                    }
                
                # Handle ultra-compact format {"p":[[team,score,"reason"]...],"s":"ok"}
                if "p" in response_data and isinstance(response_data["p"], list):
                    logger.info(f"Response contains {len(response_data['p'])} teams in ultra-compact format")
                    
                    # Log first few teams for debugging
                    teams_sample = response_data["p"][:3]
                    logger.info(f"First few teams (ultra-compact): {json.dumps(teams_sample)}")
                    
                    # Check for repeating patterns in teams
                    team_nums = [int(entry[0]) for entry in response_data["p"] if len(entry) >= 1]
                    team_counts = {}
                    for team_num in team_nums:
                        team_counts[team_num] = team_counts.get(team_num, 0) + 1
                    
                    # Check if we have duplicates
                    duplicates = {team: count for team, count in team_counts.items() if count > 1}
                    if duplicates:
                        logger.warning(f"Response contains duplicates: {duplicates}")
                        logger.warning(f"First 20 team numbers: {team_nums[:20]}")
                        
                        # Check if we have a repeating pattern
                        if len(team_nums) > 16:
                            # Check for common sequence lengths
                            for pattern_length in [4, 8, 12, 16]:
                                if len(team_nums) >= pattern_length * 2:
                                    pattern1 = team_nums[:pattern_length]
                                    pattern2 = team_nums[pattern_length:pattern_length*2]
                                    if pattern1 == pattern2:
                                        logger.warning(f"Detected repeating pattern of length {pattern_length}")
                                        logger.warning(f"Model is repeating teams instead of ranking all teams")
                                        # Truncate to first pattern only to avoid duplicates
                                        logger.warning(f"Truncating response to first {pattern_length} teams")
                                        response_data["p"] = response_data["p"][:pattern_length]
                                        break
                    
                    # Calculate duplication percentage
                    total_entries = len(team_nums)
                    unique_teams = len(team_counts)
                    if total_entries > 0:  # Prevent division by zero
                        duplication_percentage = ((total_entries - unique_teams) / total_entries) * 100
                        logger.warning(f"Duplication percentage: {duplication_percentage:.1f}%")
                        
                        # If duplication is extreme (over 80%), warn that model might be in a loop
                        if duplication_percentage > 80 and total_entries > 30:
                            logger.error(f"MODEL APPEARS TO BE LOOPING - {duplication_percentage:.1f}% duplicates")
                            return {
                                "status": "error",
                                "message": "The model is unable to rank all missing teams at once. Please try reducing the number of teams to rank (e.g., 10-20 at a time) or simplify the priorities. The model got stuck repeating the same teams."
                            }
                    
                    # Convert ultra-compact format to standard format
                    rankings = []
                    seen_teams = set()  # Track teams we've already added
                    
                    for team_entry in response_data["p"]:
                        if len(team_entry) >= 3:  # Ensure we have at least [team, score, reason]
                            team_number = int(team_entry[0])
                            
                            # Skip if we've seen this team already
                            if team_number in seen_teams:
                                logger.info(f"Skipping duplicate team {team_number} in response")
                                continue
                                
                            seen_teams.add(team_number)
                            score = float(team_entry[1])
                            reason = team_entry[2]
                            
                            # Get team nickname from dataset if available
                            team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                            nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                            
                            rankings.append({
                                "team_number": team_number,
                                "nickname": nickname,
                                "score": score,
                                "reasoning": reason
                            })
                            
                # Handle regular compact format
                elif "missing_team_rankings" in response_data and isinstance(response_data["missing_team_rankings"], list):
                    logger.info(f"Response contains {len(response_data['missing_team_rankings'])} teams in regular format")
                    
                    # Log first few teams for debugging
                    teams_sample = response_data["missing_team_rankings"][:3]
                    logger.info(f"First few teams: {json.dumps(teams_sample)}")
                    
                    raw_rankings = response_data["missing_team_rankings"]
                    
                    # Convert compact format to standard format if needed
                    rankings = []
                    for team_entry in raw_rankings:
                        # Check if using new compact format (has "team" instead of "team_number")
                        if "team" in team_entry and "team_number" not in team_entry:
                            team_number = team_entry["team"]
                            # Get team nickname from dataset if available
                            team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                            nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                            
                            rankings.append({
                                "team_number": team_number,
                                "nickname": nickname,
                                "score": team_entry.get("score", 0.0),
                                "reasoning": team_entry.get("reason", "No reasoning provided")
                            })
                        else:
                            # Already in standard format
                            rankings.append(team_entry)
                else:
                    logger.warning("Response has no valid rankings")
                    rankings = []
                
            except json.JSONDecodeError as e:
                # Apply same JSON error recovery as in generate_picklist
                logger.error(f"JSON parse error: {e}")
                try:
                    # Try to use regex to extract team data
                    import re
                    
                    # Try to extract from ultra-compact format first
                    # Format: [teamnum,score,"reason"] in a p array
                    compact_pattern = r'\[\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*"([^"]*)"\s*\]'
                    compact_teams_extracted = re.findall(compact_pattern, response_content)
                    
                    if compact_teams_extracted:
                        logger.info(f"Extracted {len(compact_teams_extracted)} team entries from broken ultra-compact JSON")
                        
                        # Log the first few raw extractions for debugging
                        for i, team_raw in enumerate(compact_teams_extracted[:3]):
                            logger.info(f"Raw extraction {i+1} (ultra-compact): {team_raw}")
                        
                        # Also log team numbers in sequence to check for patterns in the duplicates
                        team_numbers_sequence = [int(t[0]) for t in compact_teams_extracted[:20]]
                        logger.info(f"First 20 team numbers in missing teams response: {team_numbers_sequence}")
                        
                        rankings = []
                        team_numbers_seen = set()  # Track team numbers to detect duplicates in regex extraction
                        
                        for team_match in compact_teams_extracted:
                            try:
                                team_number = int(team_match[0])
                                score = float(team_match[1])
                                reasoning = team_match[2]
                                
                                # Skip obvious duplicates during extraction
                                if team_number in team_numbers_seen:
                                    logger.info(f"Skipping duplicate team {team_number} during regex extraction")
                                    continue
                                
                                team_numbers_seen.add(team_number)
                                
                                # Get team nickname from dataset if available
                                team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                                nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                                
                                rankings.append({
                                    "team_number": team_number,
                                    "nickname": nickname,
                                    "score": score,
                                    "reasoning": reasoning
                                })
                            except Exception as team_error:
                                logger.error(f"Error parsing team data: {team_error}")
                                continue
                                
                        logger.info(f"Salvaged {len(rankings)} teams from broken response")
                    else:
                        # Try older formats if ultra-compact format not found
                        team_pattern1 = r'"team":\s*(\d+),\s*"score":\s*([\d\.]+),\s*"reason":\s*"([^"]*)"'
                        team_pattern2 = r'"team_number":\s*(\d+),\s*"nickname":\s*"([^"]*)",\s*"score":\s*([\d\.]+),\s*"reasoning":\s*"([^"]*)"'
                        
                        teams_extracted1 = re.findall(team_pattern1, response_content)
                        teams_extracted2 = re.findall(team_pattern2, response_content)
                        
                        if teams_extracted1:
                            logger.info(f"Extracted {len(teams_extracted1)} team entries from broken compact JSON")
                            
                            # Log the first few raw extractions for debugging
                            for i, team_raw in enumerate(teams_extracted1[:3]):
                                logger.info(f"Raw extraction {i+1} (compact): {team_raw}")
                            
                            rankings = []
                            team_numbers_seen = set()  # Track team numbers to detect duplicates
                            
                            for team_match in teams_extracted1:
                                try:
                                    team_number = int(team_match[0])
                                    score = float(team_match[1])
                                    reasoning = team_match[2]
                                    
                                    # Skip obvious duplicates during extraction
                                    if team_number in team_numbers_seen:
                                        logger.info(f"Skipping duplicate team {team_number} during regex extraction")
                                        continue
                                    
                                    team_numbers_seen.add(team_number)
                                    
                                    # Get team nickname from dataset if available
                                    team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                                    nickname = team_data.get("nickname", f"Team {team_number}") if team_data else f"Team {team_number}"
                                    
                                    rankings.append({
                                        "team_number": team_number,
                                        "nickname": nickname,
                                        "score": score,
                                        "reasoning": reasoning
                                    })
                                except Exception as team_error:
                                    logger.error(f"Error parsing team data: {team_error}")
                                    continue
                            
                            logger.info(f"Salvaged {len(rankings)} teams from broken response")
                        elif teams_extracted2:
                            logger.info(f"Extracted {len(teams_extracted2)} team entries from broken standard JSON")
                            
                            # Log the first few raw extractions for debugging
                            for i, team_raw in enumerate(teams_extracted2[:3]):
                                logger.info(f"Raw extraction {i+1}: {team_raw}")
                            
                            rankings = []
                            team_numbers_seen = set()  # Track team numbers to detect duplicates
                            
                            for team_match in teams_extracted2:
                                try:
                                    team_number = int(team_match[0])
                                    team_name = team_match[1]
                                    score = float(team_match[2])
                                    reasoning = team_match[3]
                                    
                                    # Skip obvious duplicates during extraction
                                    if team_number in team_numbers_seen:
                                        logger.info(f"Skipping duplicate team {team_number} during regex extraction of missing teams")
                                        continue
                                    
                                    team_numbers_seen.add(team_number)
                                    
                                    rankings.append({
                                        "team_number": team_number,
                                        "nickname": team_name,
                                        "score": score,
                                        "reasoning": reasoning
                                    })
                                except Exception as team_error:
                                    logger.error(f"Error parsing team data: {team_error}")
                                    continue
                            
                            logger.info(f"Salvaged {len(rankings)} teams from broken response")
                        else:
                            # If we couldn't extract any teams, raise error
                            logger.error("Could not extract any team data from the broken response")
                            rankings = []
                except Exception as extract_error:
                    logger.error(f"Failed to extract team data: {extract_error}")
                    rankings = []
            
            # Process the missing team rankings
            logger.info("=== Processing missing team rankings ===")
            logger.info(f"Total teams received: {len(rankings)}")
            
            # Check for duplicate teams and handle them intelligently - same logic as in generate_picklist
            team_entries = {}  # Map team numbers to their entries
            duplicates = []
            
            for team in rankings:
                team_number = team.get("team_number")
                if not team_number:
                    continue  # Skip teams without a valid team number
                
                if team_number not in team_entries:
                    # First time seeing this team
                    team_entries[team_number] = team
                else:
                    # Found a duplicate team - keep the one with the higher score
                    duplicates.append(team_number)
                    current_score = team_entries[team_number].get("score", 0)
                    new_score = team.get("score", 0)
                    
                    if new_score > current_score:
                        # This new entry has a higher score, use it instead
                        logger.info(f"Missing team {team_number} appears twice - keeping entry with higher score ({new_score} vs {current_score})")
                        team_entries[team_number] = team
            
            # Create the deduplicated rankings from the team_entries map
            deduplicated_rankings = list(team_entries.values())
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} duplicate teams in missing teams rankings: {duplicates[:10]}...")
                logger.info(f"Resolved by keeping the entry with higher score for each team")
                
                # Analyze the duplicates in more detail
                duplicate_counts = {}
                for team_num in duplicates:
                    if team_num not in duplicate_counts:
                        duplicate_counts[team_num] = 0
                    duplicate_counts[team_num] += 1
                
                # Find teams with the most duplicates
                sorted_duplicates = sorted(duplicate_counts.items(), key=lambda x: x[1], reverse=True)
                if sorted_duplicates:
                    logger.info(f"Most duplicated teams in missing teams rankings: {sorted_duplicates[:5]}")
                    
                    # Log positions of a highly duplicated team
                    if sorted_duplicates[0][1] > 1:
                        most_duplicated = sorted_duplicates[0][0]
                        positions = [i for i, team in enumerate(rankings) if team.get('team_number') == most_duplicated]
                        logger.info(f"Team {most_duplicated} appears at positions: {positions}")
            
            logger.info(f"After deduplication: {len(deduplicated_rankings)} teams")
            
            # Check if we got all the missing teams
            ranked_team_numbers = {team["team_number"] for team in deduplicated_rankings}
            still_missing = set(missing_team_numbers) - ranked_team_numbers
            
            # Log the completeness
            coverage_percent = (len(ranked_team_numbers) / len(missing_team_numbers)) * 100 if missing_team_numbers else 0
            logger.info(f"GPT coverage: {coverage_percent:.1f}% ({len(ranked_team_numbers)} of {len(missing_team_numbers)} teams)")
            
            # For any teams that are still missing, add fallbacks
            if still_missing:
                logger.warning(f"Still missing {len(still_missing)} teams")
                logger.warning(f"Missing team numbers: {sorted(list(still_missing))}")
                
                # Get avg score from the ranked teams we were able to get for better consistency
                avg_score = sum([team["score"] for team in deduplicated_rankings]) / len(deduplicated_rankings) if deduplicated_rankings else 0.1
                backup_score = max(0.1, avg_score * 0.7)  # Use 70% of avg score
                logger.info(f"Using backup score {backup_score} for still missing teams")
                
                # Add still missing teams to the rankings
                for team_number in still_missing:
                    team_data = next((t for t in teams_data if t["team_number"] == team_number), None)
                    if team_data:
                        deduplicated_rankings.append({
                            "team_number": team_number,
                            "nickname": team_data.get("nickname", f"Team {team_number}"),
                            "score": backup_score,
                            "reasoning": "Added to complete the picklist - not enough data available for detailed analysis",
                            "is_fallback": True
                        })
            
            # Assemble final result
            result = {
                "status": "success",
                "missing_team_rankings": deduplicated_rankings,
                "performance": {
                    "total_time": request_time,
                    "team_count": len(missing_team_numbers),
                    "avg_time_per_team": request_time / len(missing_team_numbers) if missing_team_numbers else 0,
                    "missing_teams": len(still_missing),
                    "duplicate_teams": len(duplicates)
                }
            }
            
            # Log completion stats
            total_time = time.time() - start_time
            logger.info(f"====== MISSING TEAMS RANKING COMPLETE ======")
            logger.info(f"Total time: {total_time:.2f}s for {len(deduplicated_rankings)} teams")
            logger.info(f"Average time per team: {(total_time / len(deduplicated_rankings) if deduplicated_rankings else 0):.2f}s")
            
            # Cache the result, replacing the "in progress" timestamp
            self._picklist_cache[cache_key] = result
            
            logger.info(f"Successfully completed missing teams ranking{request_info}")
            return result
            
        except Exception as e:
            logger.error(f"Error ranking missing teams with GPT: {str(e)}{request_info}", exc_info=True)
            
            # Clean up the in-progress flag from cache so future requests can proceed
            if cache_key in self._picklist_cache and isinstance(self._picklist_cache[cache_key], float):
                del self._picklist_cache[cache_key]
                
            return {
                "status": "error",
                "message": f"Failed to rank missing teams: {str(e)}"
            }
    
    def _create_missing_teams_system_prompt(self, pick_position: str, team_count: int) -> str:
        """
        Create a specialized system prompt for ranking missing teams.
        Uses the ultra-compact schema to optimize token usage.
        
        Args:
            pick_position: 'first', 'second', or 'third'
            team_count: Number of missing teams to rank
            
        Returns:
            System prompt string
        """
        position_context = {
            "first": "First pick teams should be overall powerhouse teams that excel in multiple areas.",
            "second": "Second pick teams should complement the first pick and address specific needs.",
            "third": "Third pick teams are more specialized, often focusing on a single critical function."
        }
        
        return f"""You are GPT-4o, an FRC pick-list strategist.

TASK
Rank ALL {team_count} unique missing teams for the {pick_position} pick of Team {{your_team_number}}.
Return MINIFIED JSON, ONE LINE, NO SPACES/NEWLINES using THIS exact shape:

{{"p":[[team,score,"≤12 words"]...],"s":"ok"}}

⚠️ CRITICAL RULES:
1. Each team MUST appear EXACTLY ONCE. NO DUPLICATES ALLOWED!
2. Include ALL {team_count} teams from MISSING_TEAM_NUMBERS.
3. Each reason must be ≤ 12 words and cite ≥ 1 metric value.
4. NO whitespaces, tabs, or line-breaks in the output.
5. If you cannot fit ALL {team_count} teams, STOP and ONLY return: {{"s":"overflow"}}

ULTRA-COMPACT STRUCTURE EXPLANATION:
- "p" is the array of team entries (replaces "missing_team_rankings")
- Each team is [team_number, score, "reason"] (array instead of object)
- "s" is status ("ok" or "overflow")

TOKEN OPTIMIZATION EXAMPLES (use different teams):
- [254, 98.3, "Strong autonomous EPA 25.7"]
- [1678, 92.1, "Excellent climb consistency 96%"]
- [3310, 87.5, "High teleop scoring average 15.2 points"]
- [118, 85.2, "Great defense rating 8.9"]

VALIDATION REQUIREMENTS:
1. Verify ALL {team_count} teams are included - CHECK CLOSELY!
2. Check for duplicated team numbers - NONE ALLOWED!
3. Verify your reasoning strings are ≤ 12 words each
4. Ensure JSON is complete, valid, and has NO WHITESPACE

Additional context: {position_context.get(pick_position, "")}

OVERFLOW HANDLING: If you cannot include all {team_count} teams, return ONLY {{"s":"overflow"}} - nothing else.
KEY REQUIREMENT: Make your rankings consistent with the examples of already-ranked teams.
"""
    
    def _create_missing_teams_user_prompt(
        self,
        your_team_number: int,
        pick_position: str,
        priorities: List[Dict[str, Any]],
        missing_teams_data: List[Dict[str, Any]],
        ranked_teams: List[Dict[str, Any]]
    ) -> str:
        """
        Create a specialized user prompt for ranking missing teams.
        Includes samples of already-ranked teams for stylistic consistency.
        
        Args:
            your_team_number: Your team's number
            pick_position: 'first', 'second', or 'third'
            priorities: List of metric priorities with weights
            missing_teams_data: Data for teams that need to be ranked
            ranked_teams: Teams that have already been ranked (for context)
            
        Returns:
            User prompt string
        """
        # Find your team's data
        your_team_info = next((team for team in self._prepare_team_data_for_gpt() if team["team_number"] == your_team_number), None)
        
        # Get top and bottom examples from ranked teams
        ranked_sample = []
        if ranked_teams and len(ranked_teams) > 2:
            # Include top, middle, and bottom ranked teams as examples
            ranked_sample.append(ranked_teams[0])  # Top team
            middle_idx = len(ranked_teams) // 2
            ranked_sample.append(ranked_teams[middle_idx])  # Middle team
            ranked_sample.append(ranked_teams[-1])  # Bottom team
        
        missing_team_numbers = [team["team_number"] for team in missing_teams_data]
        
        prompt = f"""YOUR_TEAM_PROFILE = {json.dumps(your_team_info, indent=2) if your_team_info else "{}"} 
PRIORITY_METRICS  = {json.dumps(priorities, indent=2)}
MISSING_TEAM_NUMBERS = {json.dumps(missing_team_numbers)}

ALREADY_RANKED_EXAMPLES = {json.dumps(ranked_sample, indent=2)}

⚠️ CRITICAL INSTRUCTIONS:
1. You MUST rank ONLY the teams in MISSING_TEAM_NUMBERS.
2. Each team MUST appear EXACTLY ONCE in your output. NO DUPLICATES ALLOWED!
3. Use the same metrics and style as the ALREADY_RANKED_EXAMPLES.
4. Evaluate for alliance synergy with Team {your_team_number}.
5. Maintain consistent scoring scale with ALREADY_RANKED_EXAMPLES.
6. VERIFY before submitting that you've included ALL teams in MISSING_TEAM_NUMBERS.
7. Use the ultra-compact format: {{"p":[[team,score,"reason"]...],"s":"ok"}}

VALIDATION CHECKLIST:
- ✓ Each team from MISSING_TEAM_NUMBERS appears exactly once
- ✓ No team is duplicated or missing
- ✓ Each reason is ≤ 12 words
- ✓ Metrics are cited in each reason
- ✓ JSON format is correct with no whitespace

TEAMS_TO_RANK = {json.dumps(missing_teams_data, indent=2)}

End of prompt.
"""
        return prompt
    
    def merge_and_update_picklist(
        self,
        picklist: List[Dict[str, Any]],
        user_rankings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge an existing picklist with user-defined rankings.
        
        Args:
            picklist: Original generated picklist
            user_rankings: User-modified rankings with team numbers and positions
            
        Returns:
            Updated picklist with user modifications
        """
        # Create a map of team numbers to their picklist entries
        team_map = {entry["team_number"]: entry for entry in picklist}
        
        # Create a new picklist based on user rankings
        new_picklist = []
        for ranking in user_rankings:
            team_number = ranking["team_number"]
            if team_number in team_map:
                # Add the team with original data but at the new position
                new_picklist.append(team_map[team_number])
            else:
                # Team not in original picklist, add with minimal info
                new_picklist.append({
                    "team_number": team_number,
                    "nickname": ranking.get("nickname", f"Team {team_number}"),
                    "score": ranking.get("score", 0.0),
                    "reasoning": "Manually added by user"
                })
        
        return new_picklist