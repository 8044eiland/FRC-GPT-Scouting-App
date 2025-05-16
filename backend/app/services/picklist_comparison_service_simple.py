# backend/app/services/picklist_comparison_service_simple.py

import json
import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("picklist_comparison_simple.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('picklist_comparison_simple')

class PicklistComparisonService:
    """
    Simplified service for comparing teams in a picklist without OpenAI dependencies.
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
        logger.info(f"Loading dataset from: {unified_dataset_path}")
        self.dataset = self._load_dataset()
        self.teams_data = self.dataset.get("teams", {})
        self.year = self.dataset.get("year", 2025)
        self.event_key = self.dataset.get("event_key", f"{self.year}arc")
        logger.info(f"Dataset loaded. Team count: {len(self.teams_data)}")
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the unified dataset from the JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading unified dataset: {e}")
            return {}
    
    def _get_team_data(self, team_number: int) -> Optional[Dict[str, Any]]:
        """Get a team's data by team number."""
        team_str = str(team_number)
        return self.teams_data.get(team_str)
    
    def _prepare_team_for_comparison(self, team_number: int) -> Dict[str, Any]:
        """Prepare a team's data for comparison."""
        team_data = self._get_team_data(team_number)
        if not team_data:
            logger.warning(f"Team data not found for team {team_number}")
            return {"team_number": team_number, "error": "Team data not found"}
        
        logger.info(f"Preparing comparison data for team {team_number}")
        
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
        
        logger.info(f"Team {team_number} prepared with {len(comparison_data['metrics'])} metrics")
        return comparison_data
    
    def _compare_metrics(self, team1_data: Dict[str, Any], team2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics between two teams."""
        logger.info(f"Comparing metrics between teams {team1_data['team_number']} and {team2_data['team_number']}")
        
        team1_metrics = team1_data.get("metrics", {})
        team2_metrics = team2_data.get("metrics", {})
        
        # Find all unique metrics from both teams
        all_metrics = set(team1_metrics.keys()).union(set(team2_metrics.keys()))
        logger.info(f"Found {len(all_metrics)} unique metrics to compare")
        
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
            better_team = team1_data["team_number"] if difference > 0 else team2_data["team_number"]
            if difference == 0:
                better_team = None
                
            # Determine significance of the difference
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
        
        logger.info(f"Generated comparison for {len(comparison_results)} metrics")
        return comparison_results
    
    async def compare_teams(
        self,
        team_numbers: List[int],
        chat_history: List[Dict[str, str]] = None,
        specific_question: Optional[str] = None,
        metrics_to_highlight: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare two teams and provide analysis."""
        logger.info(f"Starting team comparison for teams: {team_numbers}")
        
        if len(team_numbers) != 2:
            logger.error("Invalid number of team numbers provided")
            return {"status": "error", "message": "Exactly two team numbers must be provided for comparison"}
            
        team1_number, team2_number = team_numbers
        
        # Check cache for this specific comparison
        cache_key = f"{team1_number}_{team2_number}"
        metrics_from_cache = False
        
        # Prepare team data
        logger.info(f"Preparing data for teams {team1_number} and {team2_number}")
        team1_data = self._prepare_team_for_comparison(team1_number)
        team2_data = self._prepare_team_for_comparison(team2_number)
        
        # Check for errors in team data
        if "error" in team1_data or "error" in team2_data:
            logger.error(f"Team data error: {team1_data.get('error')} or {team2_data.get('error')}")
            return {
                "status": "error", 
                "message": f"Team data not found for one or both teams: {team1_number}, {team2_number}"
            }
            
        # Compare metrics
        if cache_key in self._comparison_cache:
            logger.info(f"Using cached metrics comparison for {cache_key}")
            metrics_comparison = self._comparison_cache[cache_key]
            metrics_from_cache = True
        else:
            logger.info("Generating new metrics comparison")
            metrics_comparison = self._compare_metrics(team1_data, team2_data)
            self._comparison_cache[cache_key] = metrics_comparison
            
        # Filter metrics if specific ones are requested
        if metrics_to_highlight:
            logger.info(f"Filtering for specific metrics: {metrics_to_highlight}")
            filtered_comparison = {k: v for k, v in metrics_comparison.items() if k in metrics_to_highlight}
            # If we found matches, use the filtered version
            if filtered_comparison:
                metrics_comparison = filtered_comparison
        
        # For simplicity, return a basic analysis (no OpenAI)
        better_team = None
        better_team_metrics = 0
        team1_better = 0
        team2_better = 0
        
        for _, data in metrics_comparison.items():
            if data["better_team"] == team1_number:
                team1_better += 1
            elif data["better_team"] == team2_number:
                team2_better += 1
                
        if team1_better > team2_better:
            better_team = team1_number
            better_team_metrics = team1_better
        elif team2_better > team1_better:
            better_team = team2_number
            better_team_metrics = team2_better
            
        analysis_text = f"Based on the metrics comparison, "
        if better_team:
            team_name = team1_data["nickname"] if better_team == team1_number else team2_data["nickname"]
            analysis_text += f"Team {better_team} ({team_name}) appears to be stronger overall, with better performance in {better_team_metrics} metrics."
        else:
            analysis_text += "both teams appear to be evenly matched across the metrics analyzed."
            
        analysis_text += f"\n\nTeam {team1_number} ({team1_data['nickname']}) is better in {team1_better} metrics."
        analysis_text += f"\nTeam {team2_number} ({team2_data['nickname']}) is better in {team2_better} metrics."
        
        if specific_question:
            analysis_text += f"\n\nRegarding your question: '{specific_question}'\n"
            analysis_text += "Since this is a simplified version of the service, we cannot provide a detailed answer to your specific question. Please check the metrics comparison for relevant information."
            
        logger.info("Comparison completed successfully")
        
        # Combine results
        return {
            "status": "success",
            "comparison_data": {
                "metrics_comparison": metrics_comparison,
                "qualitative_analysis": analysis_text,
                "unique_strengths": {
                    "team1": [f"Better in {team1_better} metrics"],
                    "team2": [f"Better in {team2_better} metrics"]
                },
                "recommendation": analysis_text.split("\n")[0],
                "chat_response": analysis_text
            },
            "metrics_from_cache": metrics_from_cache,
            "team1": {"number": team1_number, "nickname": team1_data["nickname"]},
            "team2": {"number": team2_number, "nickname": team2_data["nickname"]},
        }