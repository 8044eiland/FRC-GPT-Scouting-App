    async def _create_missing_teams_user_prompt(
        self,
        your_team_number: int,
        pick_position: str,
        priorities: List[Dict[str, Any]],
        teams_data: List[Dict[str, Any]],
        ranked_teams: List[Dict[str, Any]],
        use_vector_search: bool = True
    ) -> str:
        """
        Create a specialized user prompt for ranking missing teams.
        
        Args:
            your_team_number: Your team's number
            pick_position: 'first', 'second', or 'third'
            priorities: List of metric priorities with weights
            teams_data: Prepared team data for missing teams
            ranked_teams: Teams that have already been ranked
            use_vector_search: Whether to use vector search for relevant context
            
        Returns:
            User prompt string
        """
        # Find your team's data
        your_team_info = self._get_team_by_number(your_team_number)
        
        # Get missing team numbers to verify completion
        missing_team_numbers = [team["team_number"] for team in teams_data]
        
        # Get the top N ranked teams for context (at most 20 to avoid token bloat)
        top_ranked = ranked_teams[:min(20, len(ranked_teams))]
        
        # Get relevant game context using vector search if enabled
        game_context = ""
        if use_vector_search:
            try:
                logger.info("Using vector search for relevant game context (missing teams)")
                game_context = await self._get_relevant_game_context(priorities)
            except Exception as e:
                logger.error(f"Error getting relevant game context for missing teams: {e}")
                # Fall back to the full context
                game_context = self.game_context
        else:
            # Use the traditional full context
            game_context = self.game_context
        
        # Create the prompt
        prompt = f"""
YOUR_TEAM_PROFILE = {json.dumps(your_team_info) if your_team_info else "{}"} 
PRIORITY_METRICS  = {json.dumps(priorities)}
GAME_CONTEXT      = {json.dumps(game_context) if game_context else "null"}
MISSING_TEAM_NUMBERS = {json.dumps(missing_team_numbers)}

INSTRUCTIONS:
1. You must rank {len(teams_data)} teams that are MISSING from the picklist.
2. Each team must appear EXACTLY ONCE in your output. NO duplicates!
3. Insert the teams based on their strength relative to the already ranked teams.
4. Use the ALREADY_RANKED_TEAMS list as a reference for ranking quality.

ALREADY_RANKED_TEAMS = {json.dumps(top_ranked)}

TEAMS_TO_RANK = {json.dumps(teams_data)}

Return a JSON object with a "p" array using this ultra-compact format:
{{"p":[[team,score,"≤12 words"]...],"s":"ok"}}

The "p" array should contain only the MISSING_TEAM_NUMBERS teams, NOT the ALREADY_RANKED_TEAMS.
"""
        return prompt