# backend/app/services/unified_event_data_service.py

import os
import json
import uuid
from typing import List, Dict, Any, Optional
import asyncio
from app.services.progress_tracker import ProgressTracker

from app.services.schema_loader import load_schemas
from app.services.scouting_parser import parse_scouting_row
# Use the enhanced parser that preserves field categories
from app.services.superscout_parser_enhanced import parse_superscout_row
from app.services.tba_client import get_event_teams, get_event_matches, get_event_rankings
from app.services.statbotics_client import get_team_epa
from app.services.sheets_service import get_sheet_values

def get_unified_dataset_path(event_key: str) -> str:
    """
    Get the path for a unified dataset based on the event key.
    
    Args:
        event_key: The event key, e.g., "2025arc"
        
    Returns:
        str: The full path to the unified dataset JSON file
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    return os.path.join(data_dir, f"unified_event_{event_key}.json")

def sanitize_for_json(obj: Any) -> Any:
    """
    Sanitize values for safe JSON serialization.
    
    Args:
        obj: Any Python object to sanitize
        
    Returns:
        The sanitized object with None values replaced with empty strings
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif obj is None:
        return ""
    else:
        return obj

async def build_unified_dataset(
    event_key: str,
    year: int,
    force_rebuild: bool = False,
    scouting_tab: str = "Scouting",
    superscout_tab: str = "SuperScouting",
    operation_id: Optional[str] = None
) -> str:
    """
    Build a unified dataset combining scouting data, TBA data, and Statbotics data.

    Args:
        event_key: The event key, e.g., "2025arc"
        year: The season year, e.g., 2025
        force_rebuild: Whether to rebuild even if dataset already exists
        scouting_tab: Name of the scouting tab in Google Sheets
        superscout_tab: Name of the superscouting tab in Google Sheets
        operation_id: Optional unique identifier for progress tracking

    Returns:
        str: Path to the unified dataset JSON file
    """
    # Check if there's an active sheet configuration in the database
    spreadsheet_id = None
    db = None
    try:
        from app.database.db import get_db_session
        from app.services.sheet_config_service import get_active_configuration

        db = next(get_db_session())
        config_result = await get_active_configuration(db, event_key, year)

        if config_result["status"] == "success":
            config = config_result["configuration"]
            # Get the spreadsheet ID from configuration
            spreadsheet_id = config["spreadsheet_id"]
            # Override tab names with configured values if available
            scouting_tab = config["match_scouting_sheet"] or scouting_tab
            if config["super_scouting_sheet"]:
                superscout_tab = config["super_scouting_sheet"]

            print(f"\U0001F535 Using configured tabs from database: scouting={scouting_tab}, superscout={superscout_tab}")
            print(f"\U0001F535 Using spreadsheet ID from configuration: {spreadsheet_id}")
    except Exception as e:
        print(f"\U0001F534 Error getting sheet configuration, using defaults: {str(e)}")
    # Initialize progress tracking if operation_id is provided
    tracker = None
    if operation_id:
        tracker = ProgressTracker.create_tracker(operation_id)
        tracker.update(1, f"Building unified dataset for {event_key} ({year})", "Initializing")
    
    print(f"\U0001F535 Loading schemas for {year}")
    load_schemas(year)
    if tracker:
        tracker.update(5, "Loaded schemas", "Load schemas")
    
    # Check if dataset already exists and if we should rebuild
    output_path = get_unified_dataset_path(event_key)
    if os.path.exists(output_path) and not force_rebuild:
        print(f"\u2705 Using existing unified dataset: {output_path}")
        if tracker:
            tracker.complete(f"Using existing unified dataset: {output_path}")
        return output_path

    # 1. Parse Match Scouting Data
    print("\U0001F535 Fetching Match Scouting data...")
    if tracker:
        tracker.update(10, "Fetching match scouting data", "Fetch scouting data")

    # First check if the tab exists
    try:
        # Check available tabs - make sure to pass the spreadsheet_id
        from app.services.sheets_service import get_all_sheet_names
        tabs_result = await get_all_sheet_names(spreadsheet_id, db)
        available_tabs = tabs_result.get("sheet_names", []) if tabs_result.get("status") == "success" else []

        print(f"\U0001F535 Available tabs in Google Sheet: {available_tabs}")

        if scouting_tab not in available_tabs:
            print(f"\U0001F534 WARNING: '{scouting_tab}' tab not found in Google Sheet")
            scouting_raw = []
        else:
            # Use a more specific range to avoid potential API issues
            # Make sure to pass the spreadsheet_id
            scouting_raw = await get_sheet_values(f"{scouting_tab}!A1:ZZ1000", spreadsheet_id, db)
    except Exception as e:
        print(f"\U0001F534 ERROR checking available tabs: {str(e)}")
        # Try to fetch the data anyway - with spreadsheet_id
        scouting_raw = await get_sheet_values(f"{scouting_tab}!A1:ZZ1000", spreadsheet_id, db)

    # Enhanced logging to debug scouting data issues
    print(f"\U0001F535 Scouting raw data received: {len(scouting_raw) if scouting_raw else 0} rows")

    # Check for empty data
    if not scouting_raw:
        print("\U0001F534 ERROR: No scouting data received from Google Sheets")

    headers = scouting_raw[0] if scouting_raw else []
    scouting_rows = scouting_raw[1:] if scouting_raw else []

    print(f"\U0001F535 Scouting headers: {len(headers)} columns")
    print(f"\U0001F535 Scouting data rows: {len(scouting_rows)} rows")

    if tracker:
        tracker.update(15, f"Processing {len(scouting_rows)} scouting records", "Process scouting data")
    
    scouting_parsed = []
    for i, row in enumerate(scouting_rows):
        # Update progress every 10 rows to avoid too many updates
        if tracker and i % 10 == 0 and scouting_rows:
            progress_pct = 15 + (5 * i / len(scouting_rows))
            tracker.update(progress_pct, f"Processed {i}/{len(scouting_rows)} scouting records", "Process scouting data")
            
        parsed = parse_scouting_row(row, headers)

        # Debug logging for scouting parser
        if i == 0:  # Log the first row parsing for debugging
            print(f"\U0001F535 Sample row: {row[:5]}...")
            print(f"\U0001F535 Parsed result: {parsed}")

            # Extra debug for critical fields
            if not parsed or "team_number" not in parsed:
                print(f"\U0001F534 ERROR: team_number not found in parsed data!")
                print(f"\U0001F534 Headers: {headers[:5]}...")
                # Check schema mapping
                from app.services.schema_loader import get_match_mapping
                match_mapping = get_match_mapping()
                print(f"\U0001F534 Schema mapping: {match_mapping}")

        if parsed:
            # Ensure match_number is present and is an integer
            if "match_number" in parsed and parsed["match_number"] is not None:
                try:
                    parsed["match_number"] = int(parsed["match_number"])
                except (ValueError, TypeError):
                    pass  # Keep as is if conversion fails
            if "qual_number" in parsed and parsed["qual_number"] is not None:
                try:
                    parsed["qual_number"] = int(parsed["qual_number"])
                except (ValueError, TypeError):
                    pass  # Keep as is if conversion fails

            # Ensure team_number is an integer
            if "team_number" in parsed and parsed["team_number"] is not None:
                try:
                    parsed["team_number"] = int(parsed["team_number"])
                except (ValueError, TypeError):
                    pass  # Keep as is if conversion fails

            scouting_parsed.append(parsed)
    
    if tracker:
        tracker.update(20, f"Completed processing {len(scouting_parsed)} scouting records", "Process scouting data")

    # 2. Parse SuperScouting Data
    print("\U0001F535 Fetching SuperScouting data...")
    if tracker:
        tracker.update(25, "Fetching superscouting data", "Fetch superscouting data")

    # Check if the SuperScouting tab exists
    try:
        # Re-use available_tabs from before if it was set
        if 'available_tabs' not in locals():
            from app.services.sheets_service import get_all_sheet_names
            tabs_result = await get_all_sheet_names(spreadsheet_id, db)
            available_tabs = tabs_result.get("sheet_names", []) if tabs_result.get("status") == "success" else []
            print(f"\U0001F535 Available tabs in Google Sheet: {available_tabs}")

        if superscout_tab not in available_tabs:
            print(f"\U0001F534 WARNING: '{superscout_tab}' tab not found in Google Sheet")
            superscouting_raw = []
        else:
            # Use a more specific range to avoid potential API issues
            # Make sure to pass the spreadsheet_id
            superscouting_raw = await get_sheet_values(f"{superscout_tab}!A1:ZZ1000", spreadsheet_id, db)
    except Exception as e:
        print(f"\U0001F534 ERROR checking available tabs for superscouting: {str(e)}")
        # Try to fetch the data anyway - with spreadsheet_id
        superscouting_raw = await get_sheet_values(f"{superscout_tab}!A1:ZZ1000", spreadsheet_id, db)

    # Enhanced logging to debug superscouting data issues
    print(f"\U0001F535 SuperScouting raw data received: {len(superscouting_raw) if superscouting_raw else 0} rows")

    # Check for empty data
    if not superscouting_raw:
        print("\U0001F534 ERROR: No superscouting data received from Google Sheets")

    superscouting_headers = superscouting_raw[0] if superscouting_raw else []
    superscouting_rows = superscouting_raw[1:] if superscouting_raw else []

    print(f"\U0001F535 SuperScouting headers: {len(superscouting_headers)} columns")
    print(f"\U0001F535 SuperScouting data rows: {len(superscouting_rows)} rows")

    if tracker:
        tracker.update(30, f"Processing {len(superscouting_rows)} superscouting records", "Process superscouting data")
    
    superscouting_parsed = []
    for i, row in enumerate(superscouting_rows):
        # Update progress every 10 rows to avoid too many updates
        if tracker and i % 10 == 0 and superscouting_rows:
            progress_pct = 30 + (5 * i / len(superscouting_rows))
            tracker.update(progress_pct, f"Processed {i}/{len(superscouting_rows)} superscouting records", "Process superscouting data")
            
        parsed_robots = parse_superscout_row(row, superscouting_headers)
        
        # Each row can generate multiple robot entries due to robot grouping
        for robot_data in parsed_robots:
            # Only include entries with valid team numbers
            if "team_number" in robot_data and robot_data["team_number"]:
                # Ensure match_number is present if available
                if "match_number" in robot_data and robot_data["match_number"] is not None:
                    try:
                        robot_data["match_number"] = int(robot_data["match_number"])
                    except (ValueError, TypeError):
                        pass
                        
                # Ensure qual_number is present if available
                if "qual_number" in robot_data and robot_data["qual_number"] is not None:
                    try:
                        robot_data["qual_number"] = int(robot_data["qual_number"])
                    except (ValueError, TypeError):
                        pass
                
                # Add to parsed data
                superscouting_parsed.append(robot_data)
    
    if tracker:
        tracker.update(35, f"Completed processing {len(superscouting_parsed)} superscouting records", "Process superscouting data")

    # 3. Pull TBA Data
    print("\U0001F535 Fetching TBA Teams, Matches, Rankings...")
    if tracker:
        tracker.update(40, "Fetching data from The Blue Alliance", "Fetch TBA data")
    
    # Use asyncio.gather to fetch data concurrently
    event_teams, event_matches, event_rankings = await asyncio.gather(
        get_event_teams(event_key),
        get_event_matches(event_key),
        get_event_rankings(event_key)
    )
    
    if tracker:
        tracker.update(50, f"Retrieved data for {len(event_teams)} teams and {len(event_matches)} matches", "Process TBA data")

    # 3.5 Build Expected Match List
    expected_matches = []

    for match in event_matches:
        if match.get("comp_level") == "qm":  # Only qualification matches
            match_number = match.get("match_number")
            alliances = match.get("alliances", {})

            for color in ["blue", "red"]:
                teams = alliances.get(color, {}).get("team_keys", [])
                for team_key in teams:
                    if team_key.startswith("frc"):
                        team_number = int(team_key[3:])
                        expected_matches.append({
                            "match_number": match_number,
                            "team_number": team_number,
                            "alliance_color": color
                        })
    
    # Debug info about expected matches
    print(f"\U0001F535 Built {len(expected_matches)} expected match-team combinations")
    if tracker:
        tracker.update(55, f"Built {len(expected_matches)} expected match-team combinations", "Process match schedule")

    # 4. Pull Statbotics Data for each team
    print("\U0001F535 Fetching Statbotics EPA data...")
    if tracker:
        tracker.update(60, f"Fetching Statbotics EPA data for {len(event_teams)} teams", "Fetch Statbotics data")
        
    statbotics_data = {}
    
    # Process teams in smaller batches to show progress
    batch_size = max(1, len(event_teams) // 10)  # Split into ~10 progress updates
    for i in range(0, len(event_teams), batch_size):
        batch = event_teams[i:i+batch_size]
        
        if tracker:
            progress_pct = 60 + (10 * i / len(event_teams))
            tracker.update(progress_pct, f"Fetching team stats {i}/{len(event_teams)}", "Fetch Statbotics data")
        
        for team in batch:
            team_number = team.get("team_number")
            try:
                team_stats = get_team_epa(team_number, year)
                if team_stats:
                    statbotics_data[team_number] = team_stats
            except Exception as e:
                print(f"Error fetching Statbotics data for team {team_number}: {e}")
    
    if tracker:
        tracker.update(70, f"Retrieved Statbotics data for {len(statbotics_data)} teams", "Process Statbotics data")

    # 5. Merge all data
    print("\U0001F7E2 Merging datasets...")
    if tracker:
        tracker.update(75, "Merging all datasets", "Merge data")
        
    unified_data = {}
    
    # Process teams in smaller batches to show progress
    batch_size = max(1, len(event_teams) // 5)  # Split into ~5 progress updates
    for i in range(0, len(event_teams), batch_size):
        batch = event_teams[i:i+batch_size]
        
        if tracker:
            progress_pct = 75 + (15 * i / len(event_teams))
            tracker.update(progress_pct, f"Merging team data {i}/{len(event_teams)}", "Merge data")
        
        for team in batch:
            team_number = team.get("team_number")
            team_key = f"frc{team_number}"
    
            # Get team's scouting data - ensure we're using string versions of team_number for comparison
            team_scouting = [
                r for r in scouting_parsed 
                if str(r.get("team_number")) == str(team_number)
            ]
            
            # Get team's superscouting data
            team_superscouting = [
                r for r in superscouting_parsed 
                if str(r.get("team_number")) == str(team_number)
            ]
            
            # Get team's statbotics data
            team_statbotics = statbotics_data.get(team_number, {})
            
            # Get team's ranking data
            team_ranking = next(
                (r for r in event_rankings.get("rankings", []) if r.get("team_key") == team_key), 
                {}
            ) if event_rankings else {}
            
            # Collect all fields from scouting data to ensure we preserve all mappings
            team_field_mappings = {}
            for record in team_scouting:
                if "field_types" in record:
                    # Merge all field type mappings from each record
                    team_field_mappings.update(record.get("field_types", {}))
    
            unified_data[str(team_number)] = {
                "team_number": team_number,
                "nickname": team.get("nickname"),
                "scouting_data": team_scouting,
                "superscouting_data": team_superscouting,
                "tba_info": team,
                "statbotics_info": team_statbotics,
                "ranking_info": team_ranking,
                "field_mappings": team_field_mappings  # Store field mappings for later reference
            }

    # 6. Save unified data locally
    if tracker:
        tracker.update(90, "Preparing to save unified dataset", "Save data")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Extract all scouting data for top-level structure for validation
    all_scouting_records = []
    for team_number, team_data in unified_data.items():
        # Get all scouting records, preserving all fields
        for record in team_data.get("scouting_data", []):
            # Make sure all fields are present - do not filter or modify
            all_scouting_records.append(record)

    # Extract all matches from unified data
    all_matches = []
    for team_number, team_data in unified_data.items():
        for record in team_data.get("scouting_data", []):
            if "match_number" in record:
                match_found = False
                for match in all_matches:
                    if match.get("match_number") == record.get("match_number"):
                        match_found = True
                        break

                if not match_found:
                    all_matches.append({
                        "match_number": record.get("match_number"),
                        "comp_level": "qm"  # Assuming all are qualification matches
                    })

    print(f"\U0001F535 Extracted {len(all_scouting_records)} scouting records for validation")
    print(f"\U0001F535 Extracted {len(all_matches)} unique matches for validation")

    # Extract all field types used across the dataset
    all_field_types = set()
    field_count = 0
    
    # Diagnostic to check field preservation
    sample_record = None
    if all_scouting_records and len(all_scouting_records) > 0:
        sample_record = all_scouting_records[0]
        print(f"\U0001F535 Sample scouting record has {len(sample_record)} fields")
        print(f"\U0001F535 Sample fields: {list(sample_record.keys())[:10]}...")
        
        if "field_types" in sample_record:
            print(f"\U0001F535 Field types present! Found {len(sample_record['field_types'])} mapped fields")
        else:
            print(f"\U0001F534 WARNING: field_types missing from sample record")
    
    for team_data in unified_data.values():
        if "field_mappings" in team_data:
            mappings = team_data["field_mappings"]
            all_field_types.update(mappings.values())
            field_count += len(mappings)
    
    print(f"\U0001F535 Found {len(all_field_types)} unique field types across {field_count} total field mappings")
    print(f"\U0001F535 Field types: {sorted(list(all_field_types))}")
            
    output_payload = {
        "event_key": event_key,
        "year": year,
        "expected_matches": expected_matches,
        "teams": sanitize_for_json(unified_data),
        "metadata": {
            "scouting_headers": headers,
            "superscouting_headers": superscouting_headers,
            "all_field_types": list(all_field_types),
            "created_timestamp": __import__('datetime').datetime.now().isoformat(),
        },
        # Add top-level matches and scouting arrays for validation
        "matches": all_matches,
        "scouting": all_scouting_records
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\u2705 Unified dataset saved to: {output_path}")
    print(f"\u2705 Dataset contains {len(expected_matches)} expected match-team combinations")
    print(f"\u2705 Dataset contains {len(unified_data)} teams")
    
    if tracker:
        tracker.complete(f"Unified dataset created with {len(unified_data)} teams and {len(expected_matches)} match-team combinations")

    return output_path