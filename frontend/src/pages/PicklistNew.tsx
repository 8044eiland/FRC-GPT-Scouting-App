// frontend/src/pages/PicklistNew.tsx

import React, { useState, useEffect } from 'react';
import PicklistGenerator from '../components/PicklistGenerator';
import TeamComparisonModal from '../components/TeamComparisonModal';

// Type definitions
interface Team {
  team_number: number;
  nickname: string;
  stats: Record<string, number>;
  score?: number;
  metrics_contribution?: Array<{
    id: string;
    value: number;
    weighted_value: number;
    metrics_used?: string[];
  }>;
  match_count?: number;
}

interface Metric {
  id: string;
  label: string;
  category: string;
  importance_score?: number;
  win_correlation?: number;
  variability?: number;
}

interface MetricWeight {
  id: string;
  weight: number;
  reason?: string;
}

interface ParsedStrategy {
  interpretation: string;
  parsed_metrics: MetricWeight[];
}

const PicklistNew: React.FC = () => {
  // State for dataset path and data
  const [datasetPath, setDatasetPath] = useState<string>('');
  const [dataset, setDataset] = useState<any>(null);
  const [yourTeamNumber, setYourTeamNumber] = useState<number>(() => {
    const saved = localStorage.getItem('yourTeamNumber');
    return saved ? parseInt(saved) : 0;
  });
  
  // Save team number whenever it changes
  useEffect(() => {
    localStorage.setItem('yourTeamNumber', yourTeamNumber.toString());
  }, [yourTeamNumber]);
  
  // State for metrics and priorities
  const [universalMetrics, setUniversalMetrics] = useState<Metric[]>([]);
  const [gameMetrics, setGameMetrics] = useState<Metric[]>([]);
  const [suggestedMetrics, setSuggestedMetrics] = useState<Metric[]>([]);
  const [superscoutMetrics, setSuperscoutMetrics] = useState<Metric[]>([]);
  const [pitScoutMetrics, setPitScoutMetrics] = useState<Metric[]>([]);
  const [metricsStats, setMetricsStats] = useState<Record<string, any>>({});
  
  // State for metrics tab selection
  const [metricsTab, setMetricsTab] = useState<string>('suggested');
  
  // State for metrics search/filter
  const [metricsFilter, setMetricsFilter] = useState<string>('');
  
  // State for selected priorities for each pick type - with localStorage persistence
  const [firstPickPriorities, setFirstPickPriorities] = useState<MetricWeight[]>(() => {
    const saved = localStorage.getItem('firstPickPriorities');
    return saved ? JSON.parse(saved) : [];
  });
  const [secondPickPriorities, setSecondPickPriorities] = useState<MetricWeight[]>(() => {
    const saved = localStorage.getItem('secondPickPriorities');
    return saved ? JSON.parse(saved) : [];
  });
  const [thirdPickPriorities, setThirdPickPriorities] = useState<MetricWeight[]>(() => {
    const saved = localStorage.getItem('thirdPickPriorities');
    return saved ? JSON.parse(saved) : [];
  });
  
  // State for natural language prompt
  const [strategyPrompt, setStrategyPrompt] = useState<string>(() => {
    const saved = localStorage.getItem('strategyPrompt');
    return saved ? saved : '';
  });
  const [parsedPriorities, setParsedPriorities] = useState<ParsedStrategy | null>(null);
  const [isParsingStrategy, setIsParsingStrategy] = useState<boolean>(false);
  
  // State for team rankings - separate for each pick position - with localStorage persistence
  const [firstPickRankings, setFirstPickRankings] = useState<Team[]>(() => {
    const saved = localStorage.getItem('firstPickRankings');
    return saved ? JSON.parse(saved) : [];
  });
  const [secondPickRankings, setSecondPickRankings] = useState<Team[]>(() => {
    const saved = localStorage.getItem('secondPickRankings');
    return saved ? JSON.parse(saved) : [];
  });
  const [thirdPickRankings, setThirdPickRankings] = useState<Team[]>(() => {
    const saved = localStorage.getItem('thirdPickRankings');
    return saved ? JSON.parse(saved) : [];
  });
  const [isGeneratingRankings, setIsGeneratingRankings] = useState<boolean>(false);
  
  // State for team comparison
  const [selectedTeams, setSelectedTeams] = useState<number[]>([]);
  const [showComparisonModal, setShowComparisonModal] = useState<boolean>(false);
  const [comparisonTeam1, setComparisonTeam1] = useState<Team | null>(null);
  const [comparisonTeam2, setComparisonTeam2] = useState<Team | null>(null);
  
  // State for semantic batching
  const [useSemanticBatching, setUseSemanticBatching] = useState<boolean>(() => {
    const saved = localStorage.getItem('useSemanticBatching');
    return saved ? saved === 'true' : true; // Default to true for better results
  });
  // Initialize shouldShowGenerator based on whether rankings exist for the current tab
  const [shouldShowGenerator, setShouldShowGenerator] = useState<boolean>(() => {
    // Check localStorage for existing rankings
    const activeTab = localStorage.getItem('activePicklistTab') as 'first' | 'second' | 'third' || 'first';
    const firstPickData = localStorage.getItem('firstPickRankings');
    const secondPickData = localStorage.getItem('secondPickRankings');
    const thirdPickData = localStorage.getItem('thirdPickRankings');
    
    // Return true if the current tab has saved rankings
    if (activeTab === 'first' && firstPickData && JSON.parse(firstPickData).length > 0) return true;
    if (activeTab === 'second' && secondPickData && JSON.parse(secondPickData).length > 0) return true;
    if (activeTab === 'third' && thirdPickData && JSON.parse(thirdPickData).length > 0) return true;
    
    return false;
  });
  
  // State for active tab - with localStorage persistence
  const [activeTab, setActiveTab] = useState<'first' | 'second' | 'third'>(() => {
    const saved = localStorage.getItem('activePicklistTab');
    return (saved as 'first' | 'second' | 'third') || 'first';
  });
  
  // Save active tab to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('activePicklistTab', activeTab);
  }, [activeTab]);
  
  // Save priorities to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('firstPickPriorities', JSON.stringify(firstPickPriorities));
  }, [firstPickPriorities]);
  
  useEffect(() => {
    localStorage.setItem('secondPickPriorities', JSON.stringify(secondPickPriorities));
  }, [secondPickPriorities]);
  
  useEffect(() => {
    localStorage.setItem('thirdPickPriorities', JSON.stringify(thirdPickPriorities));
  }, [thirdPickPriorities]);
  
  // Save team rankings to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('firstPickRankings', JSON.stringify(firstPickRankings));
  }, [firstPickRankings]);
  
  useEffect(() => {
    localStorage.setItem('secondPickRankings', JSON.stringify(secondPickRankings));
  }, [secondPickRankings]);
  
  useEffect(() => {
    localStorage.setItem('thirdPickRankings', JSON.stringify(thirdPickRankings));
  }, [thirdPickRankings]);
  
  // Save strategy prompt to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('strategyPrompt', strategyPrompt);
  }, [strategyPrompt]);
  
  // State for loading, error, success
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isLocking, setIsLocking] = useState<boolean>(false);
  const [hasLockedPicklist, setHasLockedPicklist] = useState<boolean>(false);
  const [picklists, setPicklists] = useState<any[]>([]);
  const [activeAllianceSelection, setActiveAllianceSelection] = useState<number | null>(null);
  const [currentEventKey, setCurrentEventKey] = useState<string>("");
  const [currentYear, setCurrentYear] = useState<number>(2025);
  
  // State for confirmation popup
  const [showConfirmClear, setShowConfirmClear] = useState<boolean>(false);
  
  // State for tracking excluded teams
  const [excludedTeams, setExcludedTeams] = useState<number[]>([]);
  // State for tracking manually excluded teams - persist in localStorage
  const [manuallyExcludedTeams, setManuallyExcludedTeams] = useState<number[]>(() => {
    const savedExclusions = localStorage.getItem('manuallyExcludedTeams');
    return savedExclusions ? JSON.parse(savedExclusions) : [];
  });
  // State for team to be excluded
  const [teamToExclude, setTeamToExclude] = useState<number | ''>('');

  // Fetch dataset path and load dataset on load
  const fetchLockedPicklists = async () => {
    try {
      // First get current event info from setup
      const setupResponse = await fetch("http://localhost:8000/api/setup/info");
      let eventKey = "";

      if (setupResponse.ok) {
        const setupData = await setupResponse.json();

        if (setupData.status === "success" && setupData.event_key) {
          eventKey = setupData.event_key;
          // Also update the event key in state for display
          setCurrentEventKey(eventKey);

          // Update year if available
          if (setupData.year) {
            setCurrentYear(setupData.year);
          }
        }
      }

      // If we couldn't get event info, use a default
      if (!eventKey) {
        console.warn("Could not retrieve event info for picklists, using default");
        eventKey = "2025lake"; // Use a default that's likely to work
        setCurrentEventKey(eventKey); // Update state with default
      }

      const response = await fetch('http://localhost:8000/api/alliance/picklists');
      const data = await response.json();

      if (data.status === 'success') {
        setPicklists(data.picklists || []);

        // Check if there's a locked picklist for the current event and team
        const currentPicklist = data.picklists.find((p: any) =>
          p.team_number === yourTeamNumber && p.event_key === eventKey
        );

        setHasLockedPicklist(!!currentPicklist);
        
        // Check if there's an active alliance selection
        if (currentPicklist) {
          // Fetch alliance selections for this picklist
          try {
            const selectionResponse = await fetch(`http://localhost:8000/api/alliance/selection/${currentPicklist.id}`);
            const selectionData = await selectionResponse.json();
            
            if (selectionData.status === 'success' && selectionData.selection) {
              setActiveAllianceSelection(selectionData.selection.id);
            }
          } catch (err) {
            console.error('Error fetching alliance selection:', err);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching locked picklists:', err);
    }
  };
  
  useEffect(() => {
    const initializeData = async () => {
      try {
        // First, get current event info from setup
        const setupResponse = await fetch("http://localhost:8000/api/setup/info");
        let eventKey = "";
        let yearValue = 2025; // Default

        if (setupResponse.ok) {
          const setupData = await setupResponse.json();

          if (setupData.status === "success" && setupData.event_key) {
            eventKey = setupData.event_key;
            if (setupData.year) {
              yearValue = setupData.year;
            }
            console.log(`Using event key from setup: ${eventKey} (${yearValue})`);

            // Store these values in state for display
            setCurrentEventKey(eventKey);
            setCurrentYear(yearValue);
          }
        }

        // If we couldn't get event info from setup, show an error
        if (!eventKey) {
          console.warn("Could not retrieve event info from setup");
          setError("No event selected. Please go to Setup page and select an event first.");
          return;
        }

        // Now check for dataset with the current event key
        const response = await fetch(`http://localhost:8000/api/unified/status?event_key=${eventKey}&year=${yearValue}`);
        const data = await response.json();

        if (data.status === 'exists' && data.path) {
          setDatasetPath(data.path);

          // Load full dataset to get team rankings
          try {
            // Try to load dataset by event_key instead of path to avoid path encoding issues
            const response = await fetch(`http://localhost:8000/api/unified/dataset?event_key=${eventKey}`);
            if (response.ok) {
              const fullDataset = await response.json();
              setDataset(fullDataset);
              console.log("Dataset loaded successfully");
            } else {
              // Fallback to path-based loading if event_key fails
              console.log("Trying fallback dataset loading with path...");
              const pathResponse = await fetch(`http://localhost:8000/api/unified/dataset?path=${encodeURIComponent(data.path)}`);
              if (pathResponse.ok) {
                const pathDataset = await pathResponse.json();
                setDataset(pathDataset);
                console.log("Dataset loaded successfully via path");
              } else {
                console.error("Failed to load dataset via both methods:", await pathResponse.text());
              }
            }
          } catch (err) {
            console.error('Error loading full dataset:', err);
          }

          await fetchPicklistAnalysis(data.path);

          // Check for locked picklists
          await fetchLockedPicklists();
        }
      } catch (err) {
        console.error('Error checking datasets:', err);
        setError('Error checking for datasets');
      }
    };
    
    initializeData();
  }, []);

  // Fetch picklist analysis data
  const fetchPicklistAnalysis = async (path: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/picklist/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ unified_dataset_path: path })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch picklist analysis');
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setUniversalMetrics(data.universal_metrics || []);
        setGameMetrics(data.game_metrics || []);
        setSuggestedMetrics(data.suggested_metrics || []);
        setSuperscoutMetrics(data.superscout_metrics || []);
        
        // Extract pit scouting metrics from game metrics if they exist
        const pitMetrics = data.game_metrics?.filter((metric: Metric) => 
          metric.category === 'pit' || metric.id.includes('pit_')
        ) || [];
        
        // Also check for any explicit pit scouting metrics that might be provided
        if (data.pit_metrics && Array.isArray(data.pit_metrics)) {
          pitMetrics.push(...data.pit_metrics);
        }
        
        setPitScoutMetrics(pitMetrics);
        setMetricsStats(data.metrics_stats || {});
      } else {
        setError(data.message || 'Error in picklist analysis');
      }
    } catch (err) {
      console.error('Error fetching picklist analysis:', err);
      setError('Error loading picklist analysis data');
    } finally {
      setIsLoading(false);
    }
  };

  
  // Handle strategy prompt submission
  const handlePromptSubmit = async () => {
    if (!strategyPrompt.trim()) {
      setError('Please enter a strategy description');
      return;
    }
    
    setIsParsingStrategy(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/picklist/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          unified_dataset_path: datasetPath,
          strategy_prompt: strategyPrompt
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to parse strategy');
      }
      
      const data = await response.json();
      
      if (data.status === 'success' && data.parsed_priorities) {
        setParsedPriorities(data.parsed_priorities);

        // Log the entire parsed priorities object for debugging
        console.log("Parsed priorities from API:", JSON.stringify(data.parsed_priorities));

        // Create a completely new implementation to properly handle weights
        const parsedMetrics = data.parsed_priorities.parsed_metrics || [];

        if (parsedMetrics.length === 0) {
          console.warn("No metrics found in parsed priorities");
          setError("No metrics were found in your strategy description");
          return;
        }

        // Create new array of properly formatted metrics with correct weights
        const processedMetrics: MetricWeight[] = [];

        for (const metric of parsedMetrics) {
          // Skip invalid metrics
          if (!metric.id) {
            console.warn("Skipping metric without ID", metric);
            continue;
          }

          // Ensure weight is a valid number in our range of accepted values
          const validWeights = [0.5, 1.0, 1.5, 2.0, 3.0];

          // Parse weight regardless of format
          let weightValue: number;

          // Check different possible formats for the weight
          if (typeof metric.weight === 'number') {
            weightValue = metric.weight;
          } else if (typeof metric.weight === 'string') {
            // Try to parse string, removing any non-numeric characters except decimal point
            const cleaned = metric.weight.replace(/[^\d.]/g, '');
            weightValue = parseFloat(cleaned) || 1.0;
          } else {
            // Default if no valid weight
            weightValue = 1.0;
          }

          // Log the original and processed weight
          console.log(`Processing metric ${metric.id}: original weight=${metric.weight}, parsed=${weightValue}`);

          // Find closest valid weight value
          const closestWeight = validWeights.reduce((closest, current) => {
            return Math.abs(current - weightValue) < Math.abs(closest - weightValue) ? current : closest;
          }, 1.0);

          console.log(`Final weight for ${metric.id}: ${closestWeight}`);

          // Add properly processed metric to our array
          processedMetrics.push({
            id: metric.id,
            weight: closestWeight,
            reason: metric.reason || undefined
          });
        }

        // Now update the appropriate list of priorities
        const currentPriorities =
          activeTab === 'first' ? firstPickPriorities :
          activeTab === 'second' ? secondPickPriorities :
          thirdPickPriorities;

        // Create merged set avoiding duplicates
        const existingIds = new Set(currentPriorities.map(p => p.id));
        const mergedPriorities = [...currentPriorities];

        for (const metric of processedMetrics) {
          if (!existingIds.has(metric.id)) {
            mergedPriorities.push(metric);
            existingIds.add(metric.id);
          }
        }

        // Update state based on active tab
        if (activeTab === 'first') {
          console.log("Setting first pick priorities:", mergedPriorities);
          setFirstPickPriorities(mergedPriorities);
        } else if (activeTab === 'second') {
          console.log("Setting second pick priorities:", mergedPriorities);
          setSecondPickPriorities(mergedPriorities);
        } else {
          console.log("Setting third pick priorities:", mergedPriorities);
          setThirdPickPriorities(mergedPriorities);
        }
        
        setSuccessMessage('Strategy parsed and applied to priorities');
      } else {
        setError(data.message || 'Error parsing strategy description');
      }
    } catch (err) {
      console.error('Error parsing strategy prompt:', err);
      setError('Error parsing strategy description');
    } finally {
      setIsParsingStrategy(false);
    }
  };

  // Handle adding a metric to the current priority list
  const handleAddMetric = (metric: Metric) => {
    console.log(`Adding metric ${metric.id} (${getMetricLabel(metric.id)}) to ${activeTab} priorities`);

    const currentPriorities =
      activeTab === 'first' ? firstPickPriorities :
      activeTab === 'second' ? secondPickPriorities :
      thirdPickPriorities;

    // Check if this metric is already in the list
    if (currentPriorities.some(p => p.id === metric.id)) {
      console.log(`Metric ${metric.id} already in priorities, skipping`);
      return; // Skip if already added
    }

    // Default weight to 1.0
    const defaultWeight = 1.0;

    // Add to the appropriate list
    if (activeTab === 'first') {
      const newPriorities = [...firstPickPriorities, { id: metric.id, weight: defaultWeight }];
      console.log(`Adding to first pick priorities with weight ${defaultWeight}`, newPriorities);
      setFirstPickPriorities(newPriorities);
    } else if (activeTab === 'second') {
      const newPriorities = [...secondPickPriorities, { id: metric.id, weight: defaultWeight }];
      console.log(`Adding to second pick priorities with weight ${defaultWeight}`, newPriorities);
      setSecondPickPriorities(newPriorities);
    } else {
      const newPriorities = [...thirdPickPriorities, { id: metric.id, weight: defaultWeight }];
      console.log(`Adding to third pick priorities with weight ${defaultWeight}`, newPriorities);
      setThirdPickPriorities(newPriorities);
    }

    // Show feedback
    setSuccessMessage(`Added ${getMetricLabel(metric.id)} to ${activeTab} pick priorities`);
    setTimeout(() => setSuccessMessage(null), 2000);
  };
  
  // Handle adjusting the weight of a priority
  const handleWeightChange = (pickType: 'first' | 'second' | 'third', metricId: string, newWeight: number) => {
    console.log(`handleWeightChange called: pickType=${pickType}, metricId=${metricId}, newWeight=${newWeight} (type: ${typeof newWeight})`);

    // Ensure newWeight is a valid number
    if (isNaN(newWeight)) {
      console.error(`Invalid weight value: ${newWeight}`);
      newWeight = 1.0; // Default to 1.0 if invalid
    }

    // Validate against allowed weights
    const validWeights = [0.5, 1.0, 1.5, 2.0, 3.0];
    if (!validWeights.includes(newWeight)) {
      // Find closest valid weight
      newWeight = validWeights.reduce((prev, curr) =>
        Math.abs(curr - newWeight) < Math.abs(prev - newWeight) ? curr : prev,
        1.0 // Start with a default of 1.0
      );
      console.log(`Adjusted to valid weight: ${newWeight}`);
    }

    if (pickType === 'first') {
      // Log before state
      console.log('Before update - First priorities:', JSON.stringify(firstPickPriorities));

      const newPriorities = firstPickPriorities.map(p => {
        if (p.id === metricId) {
          console.log(`Updating ${p.id} weight from ${p.weight} to ${newWeight}`);
          return { ...p, weight: newWeight };
        }
        return p;
      });

      // Log after state
      console.log('After update - New priorities:', JSON.stringify(newPriorities));

      setFirstPickPriorities(newPriorities);
    } else if (pickType === 'second') {
      const newPriorities = secondPickPriorities.map(p =>
        p.id === metricId ? { ...p, weight: newWeight } : p
      );
      setSecondPickPriorities(newPriorities);
    } else if (pickType === 'third') {
      const newPriorities = thirdPickPriorities.map(p =>
        p.id === metricId ? { ...p, weight: newWeight } : p
      );
      setThirdPickPriorities(newPriorities);
    }

    // Force a render to ensure the UI updates
    setTimeout(() => {
      console.log(`Current priorities after update for ${pickType}:`,
        pickType === 'first' ? firstPickPriorities :
        pickType === 'second' ? secondPickPriorities :
        thirdPickPriorities
      );
    }, 0);
  };

  // Handle removing a priority
  const handleRemovePriority = (pickType: 'first' | 'second' | 'third', metricId: string) => {
    if (pickType === 'first') {
      setFirstPickPriorities(firstPickPriorities.filter(p => p.id !== metricId));
    } else if (pickType === 'second') {
      setSecondPickPriorities(secondPickPriorities.filter(p => p.id !== metricId));
    } else if (pickType === 'third') {
      setThirdPickPriorities(thirdPickPriorities.filter(p => p.id !== metricId));
    }
  };

  // Move a priority up in the list
  const handleMoveUp = (pickType: 'first' | 'second' | 'third', index: number) => {
    if (index === 0) return; // Already at the top
    
    if (pickType === 'first') {
      const newPriorities = [...firstPickPriorities];
      const temp = newPriorities[index];
      newPriorities[index] = newPriorities[index - 1];
      newPriorities[index - 1] = temp;
      setFirstPickPriorities(newPriorities);
    } else if (pickType === 'second') {
      const newPriorities = [...secondPickPriorities];
      const temp = newPriorities[index];
      newPriorities[index] = newPriorities[index - 1];
      newPriorities[index - 1] = temp;
      setSecondPickPriorities(newPriorities);
    } else if (pickType === 'third') {
      const newPriorities = [...thirdPickPriorities];
      const temp = newPriorities[index];
      newPriorities[index] = newPriorities[index - 1];
      newPriorities[index - 1] = temp;
      setThirdPickPriorities(newPriorities);
    }
  };

  // Move a priority down in the list
  const handleMoveDown = (pickType: 'first' | 'second' | 'third', index: number) => {
    const priorities = 
      pickType === 'first' ? firstPickPriorities :
      pickType === 'second' ? secondPickPriorities :
      thirdPickPriorities;
      
    if (index === priorities.length - 1) return; // Already at the bottom
    
    if (pickType === 'first') {
      const newPriorities = [...firstPickPriorities];
      const temp = newPriorities[index];
      newPriorities[index] = newPriorities[index + 1];
      newPriorities[index + 1] = temp;
      setFirstPickPriorities(newPriorities);
    } else if (pickType === 'second') {
      const newPriorities = [...secondPickPriorities];
      const temp = newPriorities[index];
      newPriorities[index] = newPriorities[index + 1];
      newPriorities[index + 1] = temp;
      setSecondPickPriorities(newPriorities);
    } else if (pickType === 'third') {
      const newPriorities = [...thirdPickPriorities];
      const temp = newPriorities[index];
      newPriorities[index] = newPriorities[index + 1];
      newPriorities[index + 1] = temp;
      setThirdPickPriorities(newPriorities);
    }
  };
  // Generate team rankings
  const generateRankings = async () => {
    const currentPriorities =
      activeTab === 'first' ? firstPickPriorities :
      activeTab === 'second' ? secondPickPriorities :
      thirdPickPriorities;

    if (currentPriorities.length === 0) {
      setError('Please add at least one priority before generating rankings');
      return;
    }

    if (!yourTeamNumber) {
      setError('Please enter your team number before generating rankings');
      return;
    }

    // Clear previous errors
    setError(null);

    // Update excluded teams based on pick position and store the result
    const teamsToExclude = updateExcludedTeams();

    // Add a UI message if teams are being excluded
    if (teamsToExclude && teamsToExclude.length > 0) {
      if (activeTab === 'second') {
        setSuccessMessage(`Excluding top 8 ranked teams (${teamsToExclude.length} teams) as they would be alliance captains`);
      } else if (activeTab === 'third') {
        setSuccessMessage(`Excluding alliance captains (${teamsToExclude.length} teams) for third pick analysis`);
      }
    }

    console.log("Starting picklist generation directly");

    // Set generating state to show loading indicator
    setIsGeneratingRankings(true);

    try {
      // Convert priorities to plain objects
      const simplePriorities = currentPriorities.map(priority => ({
        id: priority.id,
        weight: priority.weight,
        reason: priority.reason || null
      }));

      console.log(`Making API call to generate picklist for ${activeTab} pick, with ${simplePriorities.length} priorities`);
      console.log("Using dataset path:", datasetPath);
      console.log("Team:", yourTeamNumber);
      console.log("Excluding teams:", teamsToExclude);

      // Directly make the API call instead of relying on the component
      const response = await fetch('http://localhost:8000/api/picklist/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          unified_dataset_path: datasetPath,
          your_team_number: yourTeamNumber,
          pick_position: activeTab,
          priorities: simplePriorities,
          strategy_prompt: strategyPrompt, // Include the strategy prompt
          exclude_teams: teamsToExclude,
          use_batching: true,
          batch_size: 20,
          reference_teams_count: 3,
          reference_selection: "top_middle_bottom",
          use_semantic_batching: useSemanticBatching // Use the state value
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate picklist');
      }

      const data = await response.json();
      console.log("Picklist generation API response:", data);

      // Handle the response directly
      if (data.status === 'success') {
        // If batching is used, we'll get partial results first
        if (data.batched && data.cache_key) {
          console.log("Batch processing initiated with cache key:", data.cache_key);

          // Show a message indicating batch processing
          setSuccessMessage('Batch processing initiated. This may take a few minutes...');

          // Poll for updates periodically
          let complete = false;
          let attempts = 0;
          const maxAttempts = 30; // 5 minute maximum (10 second intervals)

          while (!complete && attempts < maxAttempts) {
            attempts++;
            await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds between polls

            try {
              const statusResponse = await fetch('http://localhost:8000/api/picklist/generate/status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cache_key: data.cache_key })
              });

              if (!statusResponse.ok) {
                console.error(`Polling error (attempt ${attempts}):`, statusResponse.status);
                continue;
              }

              const statusData = await statusResponse.json();
              console.log(`Polling update (attempt ${attempts}):`, statusData);

              // Check if processing is complete
              if (statusData.batch_processing?.processing_complete) {
                console.log("Batch processing complete!");

                if (statusData.picklist) {
                  // We have the final result
                  handlePicklistGenerated(statusData);
                  complete = true;
                  break;
                }
              } else {
                // Update progress message
                const progress = statusData.batch_processing?.progress_percentage || 0;
                setSuccessMessage(`Picklist generation in progress: ${progress}% complete...`);
              }
            } catch (pollError) {
              console.error(`Polling error (attempt ${attempts}):`, pollError);
            }
          }

          if (!complete) {
            setError('Picklist generation is taking longer than expected. Please wait or try again later.');
          }
        } else {
          // For non-batched results or immediate completion
          handlePicklistGenerated(data);
        }
      } else {
        setError(data.message || 'Error generating picklist');
      }
    } catch (err) {
      console.error('Error generating picklist:', err);
      setError(err.message || 'Error connecting to server');
    } finally {
      setIsGeneratingRankings(false);

      // Show the picklist generator to display results
      setShouldShowGenerator(true);
    }
  };
  
  // Create a sample dataset with rankings for testing
  const createSampleDataset = () => {
    // Sample teams with rankings - based on historically strong teams
    const sampleTeams: { [key: string]: any } = {};
    
    // Create 24 sample teams with rankings
    [254, 1114, 1678, 2056, 2767, 3310, 4414, 5254, // Top 8
     118, 148, 217, 1538, 2910, 3538, 4613, 6328,   // Next 8 
     33, 195, 330, 610, 1323, 2337, 2481, 4481      // Final 8
    ].forEach((teamNumber, index) => {
      sampleTeams[teamNumber.toString()] = {
        team_number: teamNumber,
        nickname: `Team ${teamNumber}`,
        ranking_info: {
          rank: index + 1 // Rank 1-24 based on array position
        }
      };
    });
    
    return {
      event_key: "2025arc",
      year: 2025,
      teams: sampleTeams
    };
  };

  // Handle excluding a team manually
  const handleExcludeTeam = (teamNumber: number) => {
    if (!teamNumber) {
      setError('Please enter a valid team number to exclude');
      return;
    }

    // Check if team is already manually excluded
    if (manuallyExcludedTeams.includes(teamNumber)) {
      setError(`Team ${teamNumber} is already excluded`);
      setTimeout(() => setError(null), 3000);
      return;
    }

    // Add to manually excluded teams
    const updatedManuallyExcluded = [...manuallyExcludedTeams, teamNumber];
    setManuallyExcludedTeams(updatedManuallyExcluded);
    
    // Save to localStorage
    localStorage.setItem('manuallyExcludedTeams', JSON.stringify(updatedManuallyExcluded));
    
    // Update the excluded teams list to include this newly excluded team
    updateExcludedTeams(updatedManuallyExcluded);
    
    // Show success message
    setSuccessMessage(`Team ${teamNumber} excluded from all picklists`);
    setTimeout(() => setSuccessMessage(null), 3000);
    
    // Clear the input field
    setTeamToExclude('');
    
    // Regenerate rankings with updated exclusions if we're showing the generator
    if (shouldShowGenerator) {
      setShouldShowGenerator(false);
      setTimeout(() => setShouldShowGenerator(true), 100);
    }
  };
  
  // Handle removing a team from the excluded list
  const handleRemoveExcludedTeam = (teamNumber: number) => {
    const updatedManuallyExcluded = manuallyExcludedTeams.filter(t => t !== teamNumber);
    setManuallyExcludedTeams(updatedManuallyExcluded);
    
    // Save to localStorage
    localStorage.setItem('manuallyExcludedTeams', JSON.stringify(updatedManuallyExcluded));
    
    // Update the excluded teams list
    updateExcludedTeams(updatedManuallyExcluded);
    
    // Show success message
    setSuccessMessage(`Team ${teamNumber} removed from exclusion list`);
    setTimeout(() => setSuccessMessage(null), 3000);
    
    // Regenerate rankings with updated exclusions if we're showing the generator
    if (shouldShowGenerator) {
      setShouldShowGenerator(false);
      setTimeout(() => setShouldShowGenerator(true), 100);
    }
  };
  
  // Clear all manual exclusions
  const handleClearAllExclusions = () => {
    setManuallyExcludedTeams([]);
    localStorage.removeItem('manuallyExcludedTeams');
    updateExcludedTeams([]);
    
    // Show success message
    setSuccessMessage('All manual team exclusions cleared');
    setTimeout(() => setSuccessMessage(null), 3000);
    
    // Regenerate rankings with updated exclusions if we're showing the generator
    if (shouldShowGenerator) {
      setShouldShowGenerator(false);
      setTimeout(() => setShouldShowGenerator(true), 100);
    }
  };

  // Update excluded teams based on pick position
  const updateExcludedTeams = (manualExclusions: number[] = manuallyExcludedTeams) => {
    let autoExcluded: number[] = [];
    
    // Check if dataset is available, otherwise use sample data
    const workingDataset = dataset && dataset.teams ? dataset : createSampleDataset();
    
    if (!workingDataset || !workingDataset.teams) {
      console.error("Dataset not loaded yet and fallback failed, cannot exclude teams");
      return manualExclusions;
    }
    
    console.log("Dataset teams count:", Object.keys(workingDataset.teams).length);
    
    // Log teams with rankings to debug
    const teamsWithRankings = Object.entries(workingDataset.teams)
      .filter(([_, teamData]: [string, any]) => teamData.ranking_info?.rank)
      .map(([teamNumberStr, teamData]: [string, any]) => ({
        teamNumber: parseInt(teamNumberStr),
        rank: teamData.ranking_info?.rank
      }));
    
    console.log("Teams with rankings:", teamsWithRankings);
    
    // For second pick, exclude top 8 ranked teams (assume they've been picked)
    if (activeTab === 'second') {
      try {
        // First check if we have ranking data
        const rankedTeams = Object.entries(workingDataset.teams)
          .filter(([_, teamData]: [string, any]) => teamData.ranking_info?.rank !== undefined)
          .map(([teamNumberStr, teamData]: [string, any]) => ({
            teamNumber: parseInt(teamNumberStr),
            rank: teamData.ranking_info?.rank || 999
          }));
        
        console.log(`Found ${rankedTeams.length} teams with ranking data`);
        
        if (rankedTeams.length === 0) {
          // Fallback: if no ranking data, exclude some arbitrary top team numbers
          // This is just for testing purposes
          autoExcluded = [254, 1114, 1678, 2056, 2767, 3310, 4414, 5254];
          console.log("No ranking data found. Using fallback exclusion list:", autoExcluded);
        } else {
          // Sort teams by rank (ascending) and get top 8
          const topTeams = rankedTeams
            .sort((a, b) => a.rank - b.rank)
            .slice(0, 8)
            .map(team => team.teamNumber);
          
          autoExcluded = topTeams;
          console.log("Excluding top 8 teams for second pick:", autoExcluded);
        }
      } catch (error) {
        console.error("Error getting top 8 teams:", error);
        // Fallback for testing
        autoExcluded = [254, 1114, 1678, 2056, 2767, 3310, 4414, 5254];
        console.log("Error occurred. Using fallback exclusion list:", autoExcluded);
      }
    }
    
    // For third pick, exclude only the top 8 alliance captains
    // We can't assume which specific other teams would be picked in first and second rounds
    if (activeTab === 'third') {
      try {
        // First check if we have ranking data
        const rankedTeams = Object.entries(workingDataset.teams)
          .filter(([_, teamData]: [string, any]) => teamData.ranking_info?.rank !== undefined)
          .map(([teamNumberStr, teamData]: [string, any]) => ({
            teamNumber: parseInt(teamNumberStr),
            rank: teamData.ranking_info?.rank || 999
          }));
        
        console.log(`Found ${rankedTeams.length} teams with ranking data`);
        
        if (rankedTeams.length === 0) {
          // Fallback: if no ranking data, exclude some arbitrary top team numbers
          // This is just for testing purposes - just the alliance captains
          autoExcluded = [254, 1114, 1678, 2056, 2767, 3310, 4414, 5254];
          console.log("No ranking data found. Using fallback exclusion list for third pick:", autoExcluded);
        } else {
          // Sort teams by rank (ascending) and get top 8 (alliance captains)
          const topTeams = rankedTeams
            .sort((a, b) => a.rank - b.rank)
            .slice(0, 8)
            .map(team => team.teamNumber);
          
          autoExcluded = topTeams;
          console.log("Excluding top 8 alliance captains for third pick:", autoExcluded);
        }
      } catch (error) {
        console.error("Error getting top 8 teams for third pick:", error);
        // Fallback for testing - just the alliance captains
        autoExcluded = [254, 1114, 1678, 2056, 2767, 3310, 4414, 5254];
        console.log("Error occurred. Using fallback exclusion list for third pick:", autoExcluded);
      }
    }
    
    // Combine automatic exclusions with manual exclusions, removing duplicates
    const combined = [...new Set([...autoExcluded, ...manualExclusions])];
    console.log("Final excluded teams:", combined);
    setExcludedTeams(combined);
    
    return combined; // Return the excluded teams
  };

  // Handle locking the picklist
  const handleLockPicklist = async () => {
    // Validate we have first and second picks
    if (!firstPickRankings.length || !secondPickRankings.length) {
      setError('You must generate both first and second pick rankings before locking your picklist.');
      return;
    }
    
    // Validate team number
    if (!yourTeamNumber) {
      setError('You must enter your team number before locking your picklist.');
      return;
    }
    
    setIsLocking(true);
    setError(null);
    
    try {
      // Gather all strategy prompts
      const allStrategyPrompts = {
        first: strategyPrompt || "", // Current prompt (we only keep the latest for now)
        // You could store separate prompts for each pick position if needed
      };
      
      const requestBody = {
        team_number: yourTeamNumber,
        event_key: currentEventKey,  // Using current event key from setup
        year: currentYear,          // Using current year from setup
        first_pick_data: {
          teams: firstPickRankings,
          analysis: null,      // Add analysis if available
          metadata: {
            priorities: firstPickPriorities
          }
        },
        second_pick_data: {
          teams: secondPickRankings,
          analysis: null,      // Add analysis if available
          metadata: {
            priorities: secondPickPriorities
          }
        },
        third_pick_data: thirdPickRankings.length ? {
          teams: thirdPickRankings,
          analysis: null,      // Add analysis if available
          metadata: {
            priorities: thirdPickPriorities
          }
        } : null,
        // Add excluded teams and strategy prompts for archiving
        // Make sure to include both automatically excluded teams and manually excluded teams
        excluded_teams: [...new Set([...excludedTeams, ...manuallyExcludedTeams])],
        strategy_prompts: allStrategyPrompts
      };
      
      const response = await fetch('http://localhost:8000/api/alliance/lock-picklist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to lock picklist');
      }
      
      const data = await response.json();
      const picklist_id = data.id;
      
      // Update state with success message
      setSuccessMessage('Picklist successfully locked! You can now proceed to the Alliance Selection screen.');
      setHasLockedPicklist(true);
      
      // Refresh the picklists
      await fetchLockedPicklists();
      
      // Create an alliance selection for this picklist
      // Get all team numbers from our rankings
      const allTeams = new Set<number>();
      
      // Add teams from all picklists
      firstPickRankings.forEach(team => allTeams.add(team.team_number));
      secondPickRankings.forEach(team => allTeams.add(team.team_number));
      if (thirdPickRankings.length) {
        thirdPickRankings.forEach(team => allTeams.add(team.team_number));
      }
      
      try {
        // Create a new alliance selection
        const createResponse = await fetch('http://localhost:8000/api/alliance/selection/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            picklist_id: picklist_id,
            event_key: currentEventKey,
            year: currentYear,
            team_list: Array.from(allTeams)
          })
        });
        
        if (createResponse.ok) {
          const createData = await createResponse.json();
          setActiveAllianceSelection(createData.id);
          console.log('Created alliance selection with ID:', createData.id);
        } else {
          console.error('Failed to create alliance selection');
        }
      } catch (createErr) {
        console.error('Error creating alliance selection:', createErr);
      }
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
      
    } catch (err: any) {
      setError(err.message || 'Error locking picklist');
      console.error('Error locking picklist:', err);
    } finally {
      setIsLocking(false);
    }
  };
  
  // Function to unlock the picklist
  const handleUnlockPicklist = async () => {
    // Find the current picklist
    if (!picklists || picklists.length === 0) {
      setError('No locked picklist found to unlock');
      return;
    }
    
    // Find the current picklist for this team and event
    const currentPicklist = picklists.find((p: any) =>
      p.team_number === yourTeamNumber && p.event_key === currentEventKey
    );
    
    if (!currentPicklist) {
      setError('No locked picklist found to unlock');
      return;
    }
    
    try {
      setIsLocking(true); // Reuse the loading state
      setError(null);
      
      const response = await fetch(`http://localhost:8000/api/alliance/picklist/${currentPicklist.id}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to unlock picklist');
      }
      
      // Update state
      setSuccessMessage('Picklist successfully unlocked! You can now make changes to your picklist.');
      setHasLockedPicklist(false);
      setActiveAllianceSelection(null);
      
      // Refresh the picklists
      await fetchLockedPicklists();
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
      
    } catch (err: any) {
      setError(err.message || 'Error unlocking picklist');
      console.error('Error unlocking picklist:', err);
    } finally {
      setIsLocking(false);
    }
  };

  // Handle picklist generation result
  const handlePicklistGenerated = (result: any) => {
    // Reset loading state
    setIsGeneratingRankings(false);

    if (result.status === 'success' && result.picklist) {
      // Store rankings in the appropriate state based on active tab
      if (activeTab === 'first') {
        setFirstPickRankings(result.picklist);
        // State update will trigger useEffect to save to localStorage
      } else if (activeTab === 'second') {
        setSecondPickRankings(result.picklist);
        // State update will trigger useEffect to save to localStorage
      } else if (activeTab === 'third') {
        setThirdPickRankings(result.picklist);
        // State update will trigger useEffect to save to localStorage
      }

      setSuccessMessage('Picklist generated successfully');
    } else if (result.message) {
      setError(result.message);
    }
  };

  // Find metric label by ID
  const getMetricLabel = (metricId: string): string => {
    // Check all metric lists
    const allMetrics = [...universalMetrics, ...gameMetrics, ...suggestedMetrics];
    const metric = allMetrics.find(m => m.id === metricId);
    return metric ? metric.label : metricId;
  };

  // Get active priorities based on current tab
  const getActivePriorities = (): MetricWeight[] => {
    if (activeTab === 'first') return firstPickPriorities;
    if (activeTab === 'second') return secondPickPriorities;
    return thirdPickPriorities;
  };
  
  // Handle team selection for comparison
  const handleTeamSelect = (teamNumber: number) => {
    setSelectedTeams(prev => {
      // If team is already selected, remove it
      if (prev.includes(teamNumber)) {
        return prev.filter(num => num !== teamNumber);
      }
      
      // If we already have 2 teams selected, replace the oldest one
      if (prev.length >= 2) {
        return [prev[1], teamNumber];
      }
      
      // Otherwise, add the team
      return [...prev, teamNumber];
    });
  };
  
  // Get the current active rankings based on tab
  const getActiveRankings = (): Team[] => {
    if (activeTab === 'first') return firstPickRankings;
    if (activeTab === 'second') return secondPickRankings;
    return thirdPickRankings;
  };
  
  // Open the comparison modal when compare button is clicked
  const handleCompareTeams = () => {
    if (selectedTeams.length !== 2) return;
    
    const rankings = getActiveRankings();
    const team1 = rankings.find(team => team.team_number === selectedTeams[0]) || null;
    const team2 = rankings.find(team => team.team_number === selectedTeams[1]) || null;
    
    if (team1 && team2) {
      setComparisonTeam1(team1);
      setComparisonTeam2(team2);
      setShowComparisonModal(true);
    } else {
      setError('Could not find both selected teams in the current rankings');
    }
  };
  
  // Swap the positions of the two teams being compared in the picklist rankings
  const handleSwapComparisonTeams = () => {
    if (comparisonTeam1 && comparisonTeam2) {
      const rankings = getActiveRankings();
      const index1 = rankings.findIndex(team => team.team_number === comparisonTeam1?.team_number);
      const index2 = rankings.findIndex(team => team.team_number === comparisonTeam2?.team_number);
      
      // If both teams are found in the rankings, swap them regardless of position
      if (index1 !== -1 && index2 !== -1) {
        const newRankings = [...rankings];
        [newRankings[index1], newRankings[index2]] = [newRankings[index2], newRankings[index1]];
        
        // Update the appropriate rankings state
        if (activeTab === 'first') {
          setFirstPickRankings(newRankings);
        } else if (activeTab === 'second') {
          setSecondPickRankings(newRankings);
        } else {
          setThirdPickRankings(newRankings);
        }
        
        // Show feedback and close the comparison modal
        setSuccessMessage(`Swapped teams ${comparisonTeam1.team_number} and ${comparisonTeam2.team_number} in rankings`);
        setShowComparisonModal(false);
      } else {
        setError('Could not locate one or both teams in the current rankings');
      }
    }
  };
  
  // Show confirmation dialog before clearing data
  const handleClearDataClick = () => {
    setShowConfirmClear(true);
  };
  
  // Function to clear all saved data after confirmation
  const handleClearAllData = () => {
    // Hide the confirmation dialog
    setShowConfirmClear(false);
    
    // Clear picklist rankings
    setFirstPickRankings([]);
    setSecondPickRankings([]);
    setThirdPickRankings([]);
    
    // Clear priorities
    setFirstPickPriorities([]);
    setSecondPickPriorities([]);
    setThirdPickPriorities([]);
    
    // Clear manually excluded teams
    setManuallyExcludedTeams([]);
    
    // Update localStorage directly for any other saved items
    localStorage.removeItem('firstPickRankings');
    localStorage.removeItem('secondPickRankings');
    localStorage.removeItem('thirdPickRankings');
    localStorage.removeItem('firstPickPriorities');
    localStorage.removeItem('secondPickPriorities');
    localStorage.removeItem('thirdPickPriorities');
    localStorage.removeItem('manuallyExcludedTeams');
    
    // We don't clear team number as that's a user preference
    
    // Reset UI state
    setShouldShowGenerator(false);
    
    // Show success message
    setSuccessMessage('All picklist data has been cleared');
    setTimeout(() => setSuccessMessage(null), 3000);
  };
  
  // Cancel the clear operation
  const handleCancelClear = () => {
    setShowConfirmClear(false);
  };

  const handleTabChange = (tab: 'first' | 'second' | 'third') => {
    // Only change tab if different from current
    if (tab !== activeTab) {
      setActiveTab(tab);
      
      // Don't reset rankings and component state when switching tabs
      // This ensures picklists persist when switching between tabs
      
      // Clear any error or success messages when switching tabs
      setError(null);
      setSuccessMessage(null);
      
      // Set shouldShowGenerator based on whether rankings exist for this tab
      const hasRankings = 
        tab === 'first' && firstPickRankings.length > 0 || 
        tab === 'second' && secondPickRankings.length > 0 || 
        tab === 'third' && thirdPickRankings.length > 0;
      
      setShouldShowGenerator(hasRankings);
      
      // Immediately update excluded teams when switching tabs
      // This will ensure the correct teams are excluded before generating rankings
      setTimeout(() => {
        // Use setTimeout to ensure activeTab state is updated
        updateExcludedTeams();
      }, 0);
    }
  };

  if (isLoading && !universalMetrics.length) {
    return <div className="flex justify-center items-center h-64">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
    </div>;
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Picklist Builder</h1>
        <button
          onClick={handleClearDataClick}
          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center"
          title="Clear all saved picklist data"
        >
          <svg className="w-5 h-5 mr-2" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
            <path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
          </svg>
          Clear All Data
        </button>
      </div>

      {/* Data Source Information Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <div className="flex items-start">
          <div className="flex-shrink-0 mt-0.5">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="ml-3 flex-1">
            <h3 className="text-md font-medium text-blue-800">Data Source Information</h3>
            <div className="mt-2 text-sm text-blue-700 grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
              <div>
                <span className="font-semibold">Event:</span> {currentEventKey ? (
                  <span className="ml-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    {currentEventKey} ({currentYear})
                  </span>
                ) : (
                  <span className="ml-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    No event selected
                  </span>
                )}
              </div>
              <div>
                <span className="font-semibold">Dataset File:</span> {datasetPath ? (
                  <span className="ml-1 text-xs font-mono break-all">{datasetPath.split('/').pop()}</span>
                ) : (
                  <span className="ml-1 text-xs text-red-600">No dataset loaded</span>
                )}
              </div>
              <div>
                <span className="font-semibold">Teams:</span> {dataset ? (
                  <span className="ml-1">{Object.keys(dataset.teams || {}).length} teams</span>
                ) : (
                  <span className="ml-1 text-red-600">No data</span>
                )}
              </div>
              <div>
                <span className="font-semibold">Matches:</span> {dataset ? (
                  <span className="ml-1">{dataset.matches?.length || 0} matches</span>
                ) : (
                  <span className="ml-1 text-red-600">No data</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Confirmation Dialog for Clearing Data */}
      {showConfirmClear && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full">
            <div className="flex items-center text-red-600 mb-4">
              <svg className="w-6 h-6 mr-2" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
                <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
              </svg>
              <h3 className="text-xl font-bold">Confirm Delete</h3>
            </div>
            
            <p className="mb-4">
              This will delete <strong>all</strong> of your picklist data including:
            </p>
            
            <ul className="list-disc pl-5 mb-4 text-gray-700">
              <li>All saved team rankings for all pick positions</li>
              <li>All selected priority metrics</li>
              <li>All manually excluded teams</li>
            </ul>
            
            <p className="mb-6 text-gray-700">
              You will need to regenerate all picklists. This action cannot be undone.
            </p>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={handleCancelClear}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                onClick={handleClearAllData}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                Delete All Data
              </button>
            </div>
          </div>
        </div>
      )}
      
      {error && (
        <div className="p-3 mb-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {successMessage && (
        <div className="p-3 mb-4 bg-green-100 text-green-700 rounded">
          {successMessage}
        </div>
      )}
      
      {!datasetPath ? (
        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-300 mb-6">
          <h2 className="text-xl font-bold text-yellow-800 mb-2">No Dataset Found</h2>
          <p className="text-yellow-700 mb-2">
            Please build a unified dataset first before using the picklist builder.
          </p>
          <a 
            href="/workflow"
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 inline-block"
          >
            Go to Workflow
          </a>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Available Metrics */}
          <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md p-4 mb-6">
            <h2 className="text-xl font-bold mb-4">Your Team Information</h2>
            <div className="mb-4">
                <label className="block font-medium mb-2">Your Team Number</label>
                <input
                type="number"
                value={yourTeamNumber || ''}
                onChange={(e) => setYourTeamNumber(parseInt(e.target.value) || 0)}
                placeholder="Enter your team number"
                className="w-full p-2 border rounded"
                />
                <p className="text-xs text-gray-500 mt-1">Required for generating rankings</p>
            </div>
            </div>
                        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
              <h2 className="text-xl font-bold mb-4">Strategy Description</h2>
              <p className="text-gray-600 mb-4">
                Describe what you're looking for in a robot partner using natural language.
              </p>
              <textarea
                value={strategyPrompt}
                onChange={(e) => setStrategyPrompt(e.target.value)}
                className="w-full p-2 border rounded h-32 mb-4"
                placeholder="Example: I need a second pick that is good at scoring on L4 and has a reliable auto routine."
              />
              <button
                onClick={handlePromptSubmit}
                disabled={isParsingStrategy || !strategyPrompt.trim()}
                className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-blue-300"
              >
                {isParsingStrategy ? 'Processing...' : 'Parse Strategy'}
              </button>
              {parsedPriorities && (
                <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                  <h3 className="font-semibold text-blue-800 mb-2">Strategy Interpretation</h3>
                  <p className="text-sm text-blue-700 mb-3">{parsedPriorities.interpretation}</p>
                  
                  {parsedPriorities.parsed_metrics.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-blue-800 mb-1">Identified Metrics:</h4>
                      <ul className="text-xs space-y-1">
                        {parsedPriorities.parsed_metrics.map((metric, idx) => (
                          <li key={idx} className="flex justify-between">
                            <span>{getMetricLabel(metric.id)}</span>
                            <span className="text-blue-600">Weight: {metric.weight.toFixed(1)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">Available Metrics</h2>
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Search metrics..."
                    value={metricsFilter}
                    onChange={(e) => setMetricsFilter(e.target.value)}
                    className="px-3 py-1.5 border rounded-lg text-sm w-60"
                  />
                  {metricsFilter && (
                    <button
                      onClick={() => setMetricsFilter('')}
                      className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                    >
                      ×
                    </button>
                  )}
                </div>
              </div>
              
              {/* Tabbed Metrics Interface */}
              <div className="mb-6">
                <div className="mb-4 border-b pb-4">
                  {/* Category Selector */}
                  <div className="inline-flex items-center relative mb-2">
                    <div className="w-3 h-3 rounded-full mr-2" 
                      style={{
                        backgroundColor: 
                          metricsTab === 'suggested' ? '#3B82F6' : // blue-500
                          metricsTab === 'universal' ? '#6B7280' : // gray-500
                          metricsTab === 'field' ? '#10B981' :     // green-500
                          metricsTab === 'super' ? '#8B5CF6' :     // purple-500
                          '#F59E0B'                                // amber-500 (pit)
                      }}>
                    </div>
                    <select
                      value={metricsTab}
                      onChange={(e) => setMetricsTab(e.target.value)}
                      className="py-2 pr-10 pl-2 font-medium text-lg bg-white border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-full md:w-auto"
                    >
                      <option value="suggested">
                        Suggested Metrics ({suggestedMetrics.length})
                      </option>
                      <option value="universal">
                        Universal Metrics ({universalMetrics.length})
                      </option>
                      <option value="field">
                        Field Scout Metrics ({gameMetrics.length})
                      </option>
                      <option value="super">
                        SuperScout Metrics ({superscoutMetrics.length})
                      </option>
                      <option value="pit" disabled={pitScoutMetrics.length === 0}>
                        Pit Scout Metrics ({pitScoutMetrics.length})
                      </option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2 text-gray-700">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                      </svg>
                    </div>
                  </div>
                  
                  {/* Description Message - Below the dropdown */}
                  <div className="text-sm text-gray-600 italic pb-1 max-w-2xl">
                    {metricsTab === 'suggested' ? (
                      "Top-ranked metrics for effective picklists"
                    ) : metricsTab === 'universal' ? (
                      "Metrics that apply to any FRC game"
                    ) : metricsTab === 'field' ? (
                      "Data collected during match scouting"
                    ) : metricsTab === 'super' ? (
                      "Qualitative assessments from experienced scouts"
                    ) : (
                      "Technical specifications from pit visits"
                    )}
                  </div>
                </div>
                
                {/* Helper function to filter metrics */}
                {(() => {
                  // Filter function for metrics based on search term
                  const filterMetrics = (metrics: Metric[]) => {
                    if (!metricsFilter) return metrics;
                    
                    const lowerCaseFilter = metricsFilter.toLowerCase();
                    return metrics.filter(metric => 
                      metric.label.toLowerCase().includes(lowerCaseFilter) || 
                      metric.id.toLowerCase().includes(lowerCaseFilter) ||
                      metric.category.toLowerCase().includes(lowerCaseFilter)
                    );
                  };
                  
                  // Variables to store filtered metrics
                  const filteredSuggested = filterMetrics(suggestedMetrics);
                  const filteredUniversal = filterMetrics(universalMetrics);
                  const filteredGame = filterMetrics(gameMetrics);
                  const filteredSuperscout = filterMetrics(superscoutMetrics);
                  const filteredPit = filterMetrics(pitScoutMetrics);
                  
                  return (
                    <>
                      {/* Empty state when no metrics match filter */}
                      {metricsFilter && 
                        !(filteredSuggested.length || filteredUniversal.length || 
                        filteredGame.length || filteredSuperscout.length || filteredPit.length) && (
                        <div className="p-4 text-center text-gray-500 border-2 border-dashed rounded">
                          <p>No metrics match your search for <strong>"{metricsFilter}"</strong></p>
                        </div>
                      )}
                      
                      {/* Suggested Metrics Tab */}
                      {metricsTab === 'suggested' && suggestedMetrics.length > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-blue-700">Suggested Metrics</h3>
                            <span className="text-xs text-gray-500">
                              These metrics are most likely to create a good picklist
                            </span>
                          </div>
                          
                          {filteredSuggested.length > 0 ? (
                            filteredSuggested.map((metric) => (
                              <div
                                key={metric.id}
                                onClick={() => handleAddMetric(metric)}
                                className="bg-blue-50 border border-blue-200 p-3 rounded mb-2 flex justify-between items-center cursor-pointer hover:bg-blue-100"
                              >
                                <div>
                                  <span className="font-medium">{metric.label}</span>
                                  <span className="ml-2 text-xs px-2 py-0.5 bg-blue-100 text-blue-800 rounded-full">
                                    {metric.category}
                                  </span>
                                </div>
                                {metric.importance_score !== undefined && (
                                  <span className="text-sm text-gray-600">
                                    Score: {metric.importance_score.toFixed(2)}
                                  </span>
                                )}
                              </div>
                            ))
                          ) : metricsFilter ? (
                            <div className="p-3 text-center text-gray-500 border-dashed border rounded">
                              <p>No suggested metrics match your search</p>
                            </div>
                          ) : null}
                        </div>
                      )}
                      
                      {/* Universal Metrics Tab */}
                      {metricsTab === 'universal' && universalMetrics.length > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-gray-700">Universal Metrics</h3>
                            <span className="text-xs text-gray-500">
                              These metrics apply to any FRC game and season
                            </span>
                          </div>
                          
                          {filteredUniversal.length > 0 ? (
                            filteredUniversal.map((metric) => (
                              <div
                                key={metric.id}
                                onClick={() => handleAddMetric(metric)}
                                className="bg-white border p-3 rounded mb-2 cursor-pointer hover:bg-gray-50"
                              >
                                <span className="font-medium">{metric.label}</span>
                                <span className="ml-2 text-xs px-2 py-0.5 bg-gray-100 text-gray-800 rounded-full">
                                  {metric.category}
                                </span>
                              </div>
                            ))
                          ) : metricsFilter ? (
                            <div className="p-3 text-center text-gray-500 border-dashed border rounded">
                              <p>No universal metrics match your search</p>
                            </div>
                          ) : null}
                        </div>
                      )}
                      
                      {/* Field Scout Metrics Tab */}
                      {metricsTab === 'field' && gameMetrics.length > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-green-700">Field Scout Metrics</h3>
                            <span className="text-xs text-gray-500">
                              Game-specific metrics from field scouting data
                            </span>
                          </div>
                          
                          {filteredGame.length > 0 ? (
                            filteredGame.map((metric) => (
                              <div
                                key={metric.id}
                                onClick={() => handleAddMetric(metric)}
                                className="bg-green-50 border border-green-200 p-3 rounded mb-2 cursor-pointer hover:bg-green-100"
                              >
                                <span className="font-medium">{metric.label}</span>
                                <span className="ml-2 text-xs px-2 py-0.5 bg-green-100 text-green-800 rounded-full">
                                  {metric.category}
                                </span>
                              </div>
                            ))
                          ) : metricsFilter ? (
                            <div className="p-3 text-center text-gray-500 border-dashed border rounded">
                              <p>No field scout metrics match your search</p>
                            </div>
                          ) : null}
                        </div>
                      )}
                      
                      {/* SuperScouting Metrics Tab */}
                      {metricsTab === 'super' && superscoutMetrics.length > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-purple-700">SuperScouting Metrics</h3>
                            <span className="text-xs text-gray-500">
                              Qualitative metrics from experienced scouts
                            </span>
                          </div>
                          
                          {filteredSuperscout.length > 0 ? (
                            filteredSuperscout.map((metric) => (
                              <div
                                key={metric.id}
                                onClick={() => handleAddMetric(metric)}
                                className="bg-purple-50 border border-purple-200 p-3 rounded mb-2 cursor-pointer hover:bg-purple-100"
                              >
                                <span className="font-medium">{metric.label}</span>
                                <span className="ml-2 text-xs px-2 py-0.5 bg-purple-100 text-purple-800 rounded-full">
                                  {metric.category}
                                </span>
                              </div>
                            ))
                          ) : metricsFilter ? (
                            <div className="p-3 text-center text-gray-500 border-dashed border rounded">
                              <p>No super scout metrics match your search</p>
                            </div>
                          ) : null}
                        </div>
                      )}
                      
                      {/* Pit Scout Metrics Tab */}
                      {metricsTab === 'pit' && pitScoutMetrics.length > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-amber-700">Pit Scouting Metrics</h3>
                            <span className="text-xs text-gray-500">
                              Technical specifications from pit visits
                            </span>
                          </div>
                          
                          {filteredPit.length > 0 ? (
                            filteredPit.map((metric) => (
                              <div
                                key={metric.id}
                                onClick={() => handleAddMetric(metric)}
                                className="bg-amber-50 border border-amber-200 p-3 rounded mb-2 cursor-pointer hover:bg-amber-100"
                              >
                                <span className="font-medium">{metric.label}</span>
                                <span className="ml-2 text-xs px-2 py-0.5 bg-amber-100 text-amber-800 rounded-full">
                                  {metric.category}
                                </span>
                              </div>
                            ))
                          ) : metricsFilter ? (
                            <div className="p-3 text-center text-gray-500 border-dashed border rounded">
                              <p>No pit scout metrics match your search</p>
                            </div>
                          ) : null}
                        </div>
                      )}
                    </>
                  );
                })()}
                
                {/* Empty State */}
                {((metricsTab === 'suggested' && suggestedMetrics.length === 0) ||
                  (metricsTab === 'universal' && universalMetrics.length === 0) ||
                  (metricsTab === 'field' && gameMetrics.length === 0) ||
                  (metricsTab === 'super' && superscoutMetrics.length === 0) ||
                  (metricsTab === 'pit' && pitScoutMetrics.length === 0)) && (
                  <div className="p-4 text-center text-gray-500 border-2 border-dashed rounded">
                    <p>No metrics available in this category.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
          {/* Right Columns - Priority Lists and Team Rankings */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-4 mb-6">
              <div className="flex border-b mb-4">
              <button
                    onClick={() => handleTabChange('first')}
                    className={`py-2 px-4 font-medium ${
                        activeTab === 'first'
                        ? 'text-blue-600 border-b-2 border-blue-600'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                    >
                    First Pick
                    </button>
                    <button
                        onClick={() => handleTabChange('second')}
                        className={`py-2 px-4 font-medium ${
                            activeTab === 'second'
                            ? 'text-blue-600 border-b-2 border-blue-600'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                        >
                        Second Pick
                        </button>

                        <button
                        onClick={() => handleTabChange('third')}
                        className={`py-2 px-4 font-medium ${
                            activeTab === 'third'
                            ? 'text-blue-600 border-b-2 border-blue-600'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                        >
                        Third Pick
                        </button>
              </div>
              
              <div className="min-h-[200px] border-2 border-dashed rounded-lg p-4">
                <h3 className="text-lg font-bold mb-4">
                  {activeTab === 'first' 
                    ? 'First Pick Priorities' 
                    : activeTab === 'second' 
                      ? 'Second Pick Priorities' 
                      : 'Third Pick Priorities'
                  }
                </h3>
                
                {getActivePriorities().length === 0 ? (
                  <div className="flex flex-col items-center justify-center text-gray-400 py-8">
                    <p>Click on metrics from the left to add to your priorities</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {getActivePriorities().map((priority, index) => (
                      <div
                        key={`priority-${priority.id}`}
                        className="bg-white border rounded p-3 flex items-center shadow-sm"
                      >
                        <div className="mr-3 font-medium">{index + 1}.</div>
                        <div className="flex-1">
                          <div className="font-medium">{getMetricLabel(priority.id)}</div>
                          {priority.reason && (
                            <div className="text-xs text-gray-500">{priority.reason}</div>
                          )}
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <div className="flex items-center">
                            <span className="mr-2 text-sm">Weight:</span>

                            {/* Replace select with individual buttons for more reliable UI */}
                            <div className="flex border rounded overflow-hidden">
                              {[0.5, 1.0, 1.5, 2.0, 3.0].map(weight => (
                                <button
                                  key={weight}
                                  type="button"
                                  onClick={() => handleWeightChange(activeTab, priority.id, weight)}
                                  className={`px-2 py-1 text-xs ${
                                    priority.weight === weight
                                      ? 'bg-blue-600 text-white font-bold'
                                      : 'bg-white text-gray-800 hover:bg-gray-100'
                                  }`}
                                >
                                  {weight}×
                                </button>
                              ))}
                            </div>
                          </div>
                          
                          <div className="flex space-x-1">
                            <button
                              onClick={() => handleMoveUp(activeTab, index)}
                              disabled={index === 0}
                              className={`p-1 rounded ${index === 0 ? 'text-gray-300' : 'text-blue-500 hover:bg-blue-50'}`}
                              title="Move up"
                            >
                              ↑
                            </button>
                            <button
                              onClick={() => handleMoveDown(activeTab, index)}
                              disabled={index === getActivePriorities().length - 1}
                              className={`p-1 rounded ${index === getActivePriorities().length - 1 ? 'text-gray-300' : 'text-blue-500 hover:bg-blue-50'}`}
                              title="Move down"
                            >
                              ↓
                            </button>
                          </div>
                          
                          <button 
                            onClick={() => handleRemovePriority(activeTab, priority.id)}
                            className="text-red-500 hover:text-red-700 p-1"
                            title="Remove priority"
                          >
                            ×
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              <div className="mt-6 flex justify-between items-center">
                <div className="flex items-center bg-blue-50 px-3 py-2 rounded-lg border border-blue-200">
                  <input
                    type="checkbox"
                    id="semanticBatching"
                    checked={useSemanticBatching}
                    onChange={(e) => {
                      setUseSemanticBatching(e.target.checked);
                      localStorage.setItem('useSemanticBatching', e.target.checked.toString());
                    }}
                    className="h-4 w-4 text-blue-600 rounded mr-2"
                  />
                  <label htmlFor="semanticBatching" className="text-sm text-blue-700 font-medium flex items-center">
                    Smart Team Batching 
                    <span className="ml-1 text-blue-500 text-xs cursor-help inline-block" title="Groups similar teams for more consistent rankings">
                      ℹ️
                    </span>
                  </label>
                </div>
                <button
                  onClick={generateRankings}
                  disabled={getActivePriorities().length === 0 || isGeneratingRankings}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-green-300"
                >
                  {isGeneratingRankings ? 'Generating...' : 'Generate Picklist'}
                </button>
              </div>
            </div>
            {/* Team Rankings Section - Now with PicklistGenerator integration */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">Picklist</h2>
                
                <div className="flex space-x-3">
                  {/* Team comparison button - only enabled when 2 teams are selected */}
                  <button
                    onClick={handleCompareTeams}
                    disabled={selectedTeams.length !== 2}
                    className={`px-4 py-2 rounded font-medium flex items-center ${
                      selectedTeams.length === 2
                        ? 'bg-green-600 text-white hover:bg-green-700'
                        : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    }`}
                    title={selectedTeams.length === 2 ? 'Compare selected teams' : 'Select exactly 2 teams to compare'}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Compare Teams {selectedTeams.length > 0 ? `(${selectedTeams.length}/2)` : ''}
                  </button>
                  
                  {/* Lock/Unlock Picklist button */}
                  {(firstPickRankings.length > 0 && secondPickRankings.length > 0) && !hasLockedPicklist && (
                    <button
                      onClick={handleLockPicklist}
                      disabled={isLocking}
                      className="px-4 py-2 rounded font-medium flex items-center bg-blue-600 text-white hover:bg-blue-700"
                    >
                      {isLocking ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></div>
                          Locking...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                          </svg>
                          Lock Picklist
                        </>
                      )}
                    </button>
                  )}
                  
                  {/* Unlock Picklist button - Only show if picklist is locked */}
                  {hasLockedPicklist && (
                    <button
                      onClick={handleUnlockPicklist}
                      disabled={isLocking}
                      className="px-4 py-2 rounded font-medium flex items-center bg-green-600 text-white hover:bg-green-700"
                    >
                      {isLocking ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></div>
                          Unlocking...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 2a5 5 0 00-5 5v3H4a2 2 0 00-2 2v5a2 2 0 002 2h12a2 2 0 002-2v-5a2 2 0 00-2-2h-1V7a5 5 0 00-5-5zm3 8V7a3 3 0 10-6 0v3h6z" clipRule="evenodd" />
                          </svg>
                          Unlock Picklist
                        </>
                      )}
                    </button>
                  )}
                  
                  {/* Show Alliance Selection button if picklist is locked */}
                  {hasLockedPicklist && (
                    <a
                      href={activeAllianceSelection ? `/alliance-selection/${activeAllianceSelection}` : '/alliance-selection'}
                      className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 font-medium flex items-center"
                    >
                      <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z" />
                      </svg>
                      Alliance Selection
                    </a>
                  )}
                
                  {/* Debug button to force update excluded teams */}
                  {(activeTab === 'second' || activeTab === 'third') && (
                    <button
                      onClick={() => {
                        const excluded = updateExcludedTeams();
                        if (excluded.length > 0) {
                          setSuccessMessage(`Updated excluded teams list: ${excluded.join(', ')}`);
                        } else {
                          setError('Failed to exclude teams. Check console for debug info.');
                        }
                      }}
                      className="px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700 text-xs"
                    >
                      Force Update Exclusions
                    </button>
                  )}
                </div>
              </div>
              
              {/* Team Exclusion UI */}
              <div className="p-3 mb-4 bg-white rounded-lg border border-gray-300">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="font-semibold text-gray-800">Exclude Teams</h3>
                  {manuallyExcludedTeams.length > 0 && (
                    <button
                      onClick={handleClearAllExclusions}
                      className="px-2 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-700"
                      title="Clear all manual exclusions"
                    >
                      Clear All
                    </button>
                  )}
                </div>
                
                {/* Manual team exclusion form */}
                <div className="flex mb-4">
                  <input
                    type="number"
                    placeholder="Enter team number to exclude"
                    className="flex-1 p-2 border rounded-l"
                    value={teamToExclude}
                    onChange={(e) => setTeamToExclude(e.target.value === '' ? '' : parseInt(e.target.value))}
                    min="1"
                  />
                  <button
                    onClick={() => typeof teamToExclude === 'number' && handleExcludeTeam(teamToExclude)}
                    disabled={teamToExclude === '' || teamToExclude <= 0}
                    className="px-4 py-2 bg-red-600 text-white rounded-r hover:bg-red-700 disabled:bg-red-300"
                  >
                    Exclude
                  </button>
                </div>
                
                {/* Manually excluded teams list */}
                {manuallyExcludedTeams.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2">Manually Excluded Teams:</h4>
                    <div className="flex flex-wrap gap-2">
                      {manuallyExcludedTeams.map((teamNumber) => (
                        <div key={teamNumber} className="px-2 py-1 bg-red-100 text-red-800 rounded-lg flex items-center text-sm">
                          <span>Team {teamNumber}</span>
                          <button
                            onClick={() => handleRemoveExcludedTeam(teamNumber)}
                            className="ml-2 text-red-600 hover:text-red-800"
                            title="Remove from exclusion list"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Automatically excluded teams info */}
                {excludedTeams.length > manuallyExcludedTeams.length && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Automatically Excluded Teams:</h4>
                    <div className="text-sm text-gray-600">
                      {activeTab === 'second' 
                        ? "Top 8 teams are automatically excluded (assumed to be alliance captains)" 
                        : activeTab === 'third'
                          ? "Alliance captains are automatically excluded" 
                          : "No teams are automatically excluded for first pick"}
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {excludedTeams
                        .filter(teamNumber => !manuallyExcludedTeams.includes(teamNumber))
                        .map((teamNumber) => (
                          <div key={teamNumber} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-lg text-sm">
                            Team {teamNumber}
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Debug information, hidden by default */}
              <div className="p-3 mb-4 bg-blue-50 text-blue-800 rounded-lg border border-blue-300 text-sm" style={{display: 'none'}}>
                <p className="font-semibold">Debug Information:</p>
                <p>Current Tab: {activeTab}</p>
                <p>Excluded Teams Count: {excludedTeams.length}</p>
                <p>Teams Being Excluded: {excludedTeams.join(', ') || 'None'}</p>
                <p className="mt-2 font-semibold">Dataset Status:</p>
                <p>Dataset Loaded: {dataset ? 'Yes' : 'No'}</p>
                <p>Team Count: {dataset ? Object.keys(dataset.teams || {}).length : 0}</p>
              </div>
              
              {shouldShowGenerator ? (
                // Use the PicklistGenerator component when rankings should be shown
                <>
                  {/* Display excluded teams info when relevant */}
                  {excludedTeams.length > 0 && (activeTab === 'second' || activeTab === 'third') && (
                    <div className="p-3 mb-4 bg-yellow-50 text-yellow-800 rounded-lg border border-yellow-300 text-sm">
                      <p className="font-semibold">
                        {activeTab === 'second' ? 
                          `Excluding top 8 alliance captains (${excludedTeams.length} teams)` : 
                          `Excluding alliance captains (${excludedTeams.length} teams)`}
                      </p>
                      <p className="mt-1">
                        {activeTab === 'second' 
                          ? 'Alliance captains would be unavailable as second picks.' 
                          : 'Alliance captains would be unavailable as third picks.'}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {excludedTeams.map(team => (
                          <span key={team} className="px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded-full text-xs">
                            {team}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                
                  {/* Custom team display with checkboxes */}
                  <div className="mb-4">
                    <h3 className="text-lg font-bold mb-3">Team Rankings</h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            <th scope="col" className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Select
                            </th>
                            <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Rank
                            </th>
                            <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Team
                            </th>
                            <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Score
                            </th>
                            <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Key Metrics
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {(activeTab === 'first' ? firstPickRankings : 
                            activeTab === 'second' ? secondPickRankings : 
                            thirdPickRankings).map((team, index) => (
                            <tr 
                              key={team.team_number}
                              className={selectedTeams.includes(team.team_number) ? 'bg-green-50' : 'hover:bg-gray-50'}
                            >
                              <td className="px-2 py-4 whitespace-nowrap">
                                <input
                                  type="checkbox"
                                  checked={selectedTeams.includes(team.team_number)}
                                  onChange={() => handleTeamSelect(team.team_number)}
                                  className="h-5 w-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                                  disabled={hasLockedPicklist}
                                />
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap">
                                <div className="text-sm font-medium text-gray-900">{index + 1}</div>
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <div>
                                    <div className="text-sm font-medium text-gray-900">
                                      {team.team_number}
                                    </div>
                                    <div className="text-sm text-gray-500">
                                      {team.nickname || `Team ${team.team_number}`}
                                    </div>
                                  </div>
                                </div>
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap">
                                <div className="text-sm text-gray-900 font-medium">
                                  {team.score?.toFixed(2) || 'N/A'}
                                </div>
                              </td>
                              <td className="px-3 py-4">
                                <div className="text-sm text-gray-900">
                                  {team.metrics_contribution?.length > 0 ? (
                                    <div className="flex flex-wrap gap-1">
                                      {team.metrics_contribution.slice(0, 3).map((metric) => (
                                        <span 
                                          key={metric.id} 
                                          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800"
                                          title={`Contribution: ${metric.weighted_value.toFixed(2)}`}
                                        >
                                          {getMetricLabel(metric.id)}
                                        </span>
                                      ))}
                                      {team.metrics_contribution.length > 3 && (
                                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                                          +{team.metrics_contribution.length - 3} more
                                        </span>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-gray-500 italic">No metrics data</span>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  
                  {/* Include the TeamComparisonModal */}
                  <TeamComparisonModal
                    isOpen={showComparisonModal}
                    onClose={() => setShowComparisonModal(false)}
                    team1={comparisonTeam1}
                    team2={comparisonTeam2}
                    datasetPath={datasetPath}
                    onSwapTeams={handleSwapComparisonTeams}
                    priorities={getActivePriorities()}
                  />
                </>
              ) : (
                // Show placeholder when rankings are not yet generated
                <div className="p-4 text-center text-gray-500">
                  <p>Your picklist will appear here after clicking "Generate Rankings".</p>
                  <p className="text-sm mt-2">
                    First, set your priorities in the section above.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PicklistNew;