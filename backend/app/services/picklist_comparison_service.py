# backend/app/services/picklist_comparison_service.py

import json
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("picklist_comparison.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('picklist_comparison')

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PicklistComparisonService:
    """
    Service for comparing teams in a picklist and providing detailed analysis
    using GPT for decision-making support.
    """
    
    # Class-level cache to improve performance for repeated comparisons
    _comparison_cache = {}
    
    def __init__(self, unified_dataset_path: str):
        """
        Initialize the comparison service with the unified dataset.
        
        Args:
            unified_dataset_path: Path to the unified dataset JSON file
        """
        self.dataset_path = unified_dataset_path
        self.dataset = self._load_dataset()
        self.teams_data = self.dataset.get("teams", {})
        self.year = self.dataset.get("year", 2025)
        self.event_key = self.dataset.get("event_key", f"{self.year}arc")
        
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the unified dataset from the JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading unified dataset: {e}")
            return {}
    
    def _get_team_data(self, team_number: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific team's data by team number.
        
        Args:
            team_number: The team number to look up
            
        Returns:
            Dictionary with team data or None if not found
        """
        team_str = str(team_number)
        return self.teams_data.get(team_str)
    
    def _prepare_team_for_comparison(self, team_number: int) -> Dict[str, Any]:
        """
        Prepare a team's data for comparison, extracting the most relevant metrics.
        
        Args:
            team_number: The team number to prepare data for
            
        Returns:
            Dictionary with formatted team data for comparison
        """
        team_data = self._get_team_data(team_number)
        if not team_data:
            return {"team_number": team_number, "error": "Team data not found"}
        
        # Create structured data for comparison
        comparison_data = {
            "team_number": team_number,
            "nickname": team_data.get("nickname", f"Team {team_number}"),
            "metrics": {},
            "match_count": len(team_data.get("scouting_data", [])) if team_data.get("scouting_data") else 0,
            "rank": team_data.get("ranking_info", {}).get("rank")
        }
        
        # Extract metrics from scouting data (averages)
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
                comparison_data["metrics"][metric] = sum(values) / len(values)
        
        # Add statbotics metrics
        statbotics_info = team_data.get("statbotics_info", {})
        for key, value in statbotics_info.items():
            if isinstance(value, (int, float)):
                comparison_data["metrics"][f"statbotics_{key}"] = value
        
        # Add superscouting data (qualitative)
        superscouting_notes = []
        for entry in team_data.get("superscouting_data", []):
            notes = {}
            if "strategy_notes" in entry and entry["strategy_notes"]:
                notes["strategy"] = entry["strategy_notes"]
            if "comments" in entry and entry["comments"]:
                notes["comments"] = entry["comments"]
                
            # Add any numeric ratings from superscouting
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in ["team_number", "match_number"]:
                    comparison_data["metrics"][f"superscout_{key}"] = value
                    
            if notes:
                superscouting_notes.append(notes)
        
        if superscouting_notes:
            comparison_data["superscouting_notes"] = superscouting_notes
        
        return comparison_data
    
    def _compare_metrics(self, team1_data: Dict[str, Any], team2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metrics between two teams and calculate differences.
        
        Args:
            team1_data: First team's data
            team2_data: Second team's data
            
        Returns:
            Dictionary with metric comparisons
        """
        team1_metrics = team1_data.get("metrics", {})
        team2_metrics = team2_data.get("metrics", {})
        
        # Find all unique metrics from both teams
        all_metrics = set(team1_metrics.keys()).union(set(team2_metrics.keys()))
        
        comparison_results = {}
        for metric in all_metrics:
            team1_value = team1_metrics.get(metric)
            team2_value = team2_metrics.get(metric)
            
            # Skip metrics that don't exist for both teams
            if team1_value is None or team2_value is None:
                continue
                
            difference = team1_value - team2_value
            
            # Determine which team is better for this metric
            # Note: This is simplistic and assumes higher values are better
            # A more sophisticated implementation would have a lookup table for metrics
            # where lower values are better (like cycle time)
            better_team = team1_data["team_number"] if difference > 0 else team2_data["team_number"]
            if difference == 0:
                better_team = None
                
            # Determine significance of the difference
            # This is a simple approach - could be improved with statistical analysis
            avg_value = (team1_value + team2_value) / 2
            rel_difference = abs(difference) / avg_value if avg_value != 0 else 0
            
            significance = "low"
            if rel_difference > 0.25:
                significance = "high"
            elif rel_difference > 0.1:
                significance = "medium"
                
            comparison_results[metric] = {
                "team1_value": team1_value,
                "team2_value": team2_value,
                "difference": difference,
                "better_team": better_team,
                "significance": significance
            }
            
        return comparison_results
    
    async def _generate_gpt_analysis(
        self, 
        team1_data: Dict[str, Any], 
        team2_data: Dict[str, Any],
        metrics_comparison: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        specific_question: Optional[str],
        priorities: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate analysis using GPT based on team comparison data.
        
        Args:
            team1_data: First team's data
            team2_data: Second team's data
            metrics_comparison: Results of metrics comparison
            chat_history: Previous chat messages
            specific_question: Optional specific question to answer
            
        Returns:
            GPT analysis results
        """
        # Format the messages for GPT
        messages = [
            {"role": "system", "content": f"""
You are an expert FRC scouting analyst helping a team compare robots for their alliance selection picklist.
You're comparing Team {team1_data['team_number']} ({team1_data.get('nickname', 'Unknown')}) and 
Team {team2_data['team_number']} ({team2_data.get('nickname', 'Unknown')}).

Current FRC game year: {self.year}
Event: {self.event_key}

Your analysis should be factual, clear, and focused on helping the team make an informed decision.
Focus on the most significant differences between teams and provide context about why certain metrics
matter. If one team has a clear advantage, explain why. If teams are evenly matched in key areas,
describe what additional factors might be considered for tie-breaking.

When asked a specific question, provide a concise, targeted answer specifically about the comparison.
Always frame your answers in terms of the two teams being compared.
"""
            }
        ]
        
        # Add priority information if available
        if priorities and len(priorities) > 0:
            priority_content = "TEAM PRIORITIES:\nThe following metrics are important to the team (in order of priority, with weights):\n"
            for i, priority in enumerate(priorities):
                # Access attributes directly instead of using .get()
                metric_id = priority.id if hasattr(priority, "id") else priority["id"] if isinstance(priority, dict) else ""
                weight = priority.weight if hasattr(priority, "weight") else priority["weight"] if isinstance(priority, dict) else 1.0
                reason = priority.reason if hasattr(priority, "reason") and priority.reason else priority.get("reason", "") if isinstance(priority, dict) else ""
                
                # Format metric name for readability
                metric_display = metric_id.replace('_', ' ').title()
                
                priority_content += f"{i+1}. {metric_display} (Weight: {weight})"
                if reason:
                    priority_content += f" - {reason}"
                priority_content += "\n"
            
            messages.append({"role": "system", "content": priority_content})
        
        # Format team data for GPT
        team1_metrics = json.dumps(team1_data.get("metrics", {}), indent=2)
        team2_metrics = json.dumps(team2_data.get("metrics", {}), indent=2)
        
        teams_context = f"""
TEAM {team1_data['team_number']} ({team1_data.get('nickname', 'Unknown')}):
Rank: {team1_data.get('rank', 'Unknown')}
Match Count: {team1_data.get('match_count', 0)}
Metrics: {team1_metrics}

TEAM {team2_data['team_number']} ({team2_data.get('nickname', 'Unknown')}):
Rank: {team2_data.get('rank', 'Unknown')}
Match Count: {team2_data.get('match_count', 0)}
Metrics: {team2_metrics}
"""

        # Add superscouting notes if available
        if team1_data.get("superscouting_notes"):
            teams_context += f"\nTEAM {team1_data['team_number']} SUPERSCOUTING NOTES:\n"
            for note in team1_data["superscouting_notes"]:
                if "strategy" in note:
                    teams_context += f"Strategy: {note['strategy']}\n"
                if "comments" in note:
                    teams_context += f"Comments: {note['comments']}\n"
                    
        if team2_data.get("superscouting_notes"):
            teams_context += f"\nTEAM {team2_data['team_number']} SUPERSCOUTING NOTES:\n"
            for note in team2_data["superscouting_notes"]:
                if "strategy" in note:
                    teams_context += f"Strategy: {note['strategy']}\n"
                if "comments" in note:
                    teams_context += f"Comments: {note['comments']}\n"

        # Format the metrics comparison
        comparison_context = "KEY METRICS COMPARISON:\n"
        significant_metrics = {k: v for k, v in metrics_comparison.items() if v["significance"] in ["medium", "high"]}
        
        for metric, data in significant_metrics.items():
            better_team = data["better_team"]
            team_name = team1_data["nickname"] if better_team == team1_data["team_number"] else team2_data["nickname"]
            team_number = better_team if better_team else "Equal"
            
            comparison_context += f"{metric}: Team {team_number} is better "
            comparison_context += f"({data['team1_value']} vs {data['team2_value']}, "
            comparison_context += f"difference: {abs(data['difference']):.2f}, "
            comparison_context += f"significance: {data['significance']})\n"
            
        # Add full context to messages
        messages.append({"role": "user", "content": f"{teams_context}\n\n{comparison_context}"})
        
        # Add chat history
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})
            
        # Add specific question if provided
        if specific_question:
            messages.append({"role": "user", "content": specific_question})
        else:
            messages.append({"role": "user", "content": "Please provide a comprehensive comparison of these two teams. Which team would be a better pick for an alliance and why?"})
            
        # Make the API call to GPT
        try:
            logger.info(f"Making OpenAI API call with {len(messages)} messages")
            # Use the same model as the rest of the application
            response = client.chat.completions.create(
                model="gpt-4o", # Use the same model as in picklist_generator_service.py for consistency
                messages=messages,
                temperature=0.3,
                max_tokens=1200
            )
            logger.info("OpenAI API call successful")
            
            # Extract the response content
            analysis = response.choices[0].message.content
            
            # Process the GPT response to extract components
            # This is a simplified approach - in production, you might want to
            # structure the GPT prompt to return JSON or have a more sophisticated parser
            unique_strengths = {
                "team1": [],
                "team2": []
            }
            
            recommendation = ""
            
            # Simple extraction of recommendation
            if "RECOMMENDATION:" in analysis:
                parts = analysis.split("RECOMMENDATION:")
                if len(parts) > 1:
                    recommendation = parts[1].strip()
            else:
                # Try to find conclusion-like statements
                lines = analysis.split("\n")
                for line in lines:
                    if any(x in line.lower() for x in ["conclusion", "summary", "overall", "in the end", "ultimately"]):
                        recommendation = line.strip()
                        break
                        
            return {
                "qualitative_analysis": analysis,
                "unique_strengths": unique_strengths,
                "recommendation": recommendation,
                "chat_response": analysis
            }
                
        except Exception as e:
            logger.error(f"Error generating GPT analysis: {e}")
            return {
                "qualitative_analysis": f"Error generating analysis: {str(e)}",
                "unique_strengths": {"team1": [], "team2": []},
                "recommendation": "Unable to generate recommendation due to an error.",
                "chat_response": f"Error: {str(e)}"
            }
    
    async def compare_teams(
        self,
        team_numbers: List[int],
        chat_history: List[Dict[str, str]] = None,
        specific_question: Optional[str] = None,
        metrics_to_highlight: Optional[List[str]] = None,
        priorities: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare two teams and provide detailed analysis.
        
        Args:
            team_numbers: List of two team numbers to compare
            chat_history: Previous chat messages for context
            specific_question: Optional specific question to answer
            metrics_to_highlight: Optional list of metrics to emphasize
            priorities: Optional list of metric priorities with weights and reasons
            
        Returns:
            Comparison results with metrics and analysis
        """
        logger.info(f"Starting team comparison for teams: {team_numbers}")
        logger.info(f"Dataset path: {self.dataset_path}")
        chat_history = chat_history or []
        if len(team_numbers) != 2:
            return {"status": "error", "message": "Exactly two team numbers must be provided for comparison"}
            
        team1_number, team2_number = team_numbers
        
        # Check cache for this specific comparison
        cache_key = f"{team1_number}_{team2_number}"
        if specific_question:
            cache_key += f"_{hash(specific_question)}"
            
        # Only use cache for metrics comparison, not for GPT analysis
        metrics_from_cache = False
        
        # Prepare team data with additional error logging
        try:
            logger.info(f"Preparing data for team {team1_number}")
            team1_data = self._prepare_team_for_comparison(team1_number)
            
            logger.info(f"Preparing data for team {team2_number}")
            team2_data = self._prepare_team_for_comparison(team2_number)
            
            # Check if we have valid data
            if not team1_data or not team2_data:
                logger.error(f"Missing team data for one or both teams")
                return {
                    "status": "error", 
                    "message": f"Failed to load team data for one or both teams: {team1_number}, {team2_number}"
                }
            
            # Check for specific error flags
            if "error" in team1_data or "error" in team2_data:
                error1 = team1_data.get("error", "Unknown error")
                error2 = team2_data.get("error", "Unknown error")
                logger.error(f"Team data errors: Team {team1_number}: {error1}, Team {team2_number}: {error2}")
                return {
                    "status": "error", 
                    "message": f"Team data errors: {team1_number} ({error1}), {team2_number} ({error2})"
                }
        except Exception as e:
            logger.error(f"Exception preparing team data: {str(e)}")
            return {
                "status": "error",
                "message": f"Error preparing team data: {str(e)}"
            }
            
        # Compare metrics
        if cache_key in self._comparison_cache and not specific_question:
            metrics_comparison = self._comparison_cache[cache_key]
            metrics_from_cache = True
        else:
            metrics_comparison = self._compare_metrics(team1_data, team2_data)
            # Only cache if it's not a specific question
            if not specific_question:
                self._comparison_cache[cache_key] = metrics_comparison
                
        # Filter metrics if specific ones are requested
        if metrics_to_highlight:
            filtered_comparison = {k: v for k, v in metrics_comparison.items() if k in metrics_to_highlight}
            # If we found matches, use the filtered version
            if filtered_comparison:
                metrics_comparison = filtered_comparison
                
        # Generate GPT analysis
        chat_history = chat_history or []
        analysis_result = await self._generate_gpt_analysis(
            team1_data,
            team2_data,
            metrics_comparison,
            chat_history,
            specific_question,
            priorities
        )
        
        # Combine results
        return {
            "status": "success",
            "comparison_data": {
                "metrics_comparison": metrics_comparison,
                "qualitative_analysis": analysis_result["qualitative_analysis"],
                "unique_strengths": analysis_result["unique_strengths"],
                "recommendation": analysis_result["recommendation"],
                "chat_response": analysis_result["chat_response"]
            },
            "metrics_from_cache": metrics_from_cache,
            "team1": {"number": team1_number, "nickname": team1_data["nickname"]},
            "team2": {"number": team2_number, "nickname": team2_data["nickname"]},
        }