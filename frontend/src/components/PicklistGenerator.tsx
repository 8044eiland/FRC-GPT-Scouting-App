// frontend/src/components/PicklistGenerator.tsx

import React, { useState, useEffect } from 'react';

interface Team {
  team_number: number;
  nickname: string;
  score: number;
  reasoning: string;
}

interface MetricPriority {
  id: string;
  weight: number;
  reason?: string;
}

interface PicklistAnalysis {
  draft_reasoning: string;
  evaluation: string;
  final_recommendations: string;
}

interface BatchProcessing {
  total_batches: number;
  current_batch: number;
  progress_percentage: number;
  processing_complete: boolean;
}

interface Performance {
  total_time: number;
  team_count: number;
  avg_time_per_team: number;
  missing_teams?: number;
  duplicate_teams?: number;
  batch_count?: number;
  batch_size?: number;
  reference_teams_count?: number;
  reference_selection?: string;
}

interface PicklistResult {
  status: string;
  picklist: Team[];
  analysis: PicklistAnalysis;
  missing_team_numbers?: number[];
  performance?: Performance;
  message?: string;
  batched?: boolean;
  batch_processing?: BatchProcessing;
  cache_key?: string;
}

interface MissingTeamsResult {
  status: string;
  missing_team_rankings: Team[];
  performance?: Performance;
  message?: string;
}

interface PicklistGeneratorProps {
  datasetPath: string;
  yourTeamNumber: number;
  pickPosition: 'first' | 'second' | 'third';
  priorities: MetricPriority[];
  strategyPrompt?: string; // Add strategy prompt prop
  excludeTeams?: number[];
  onPicklistGenerated?: (result: PicklistResult) => void;
  initialPicklist?: Team[]; // Add prop for initial picklist data
  onExcludeTeam?: (teamNumber: number) => void; // Callback for excluding a team
  isLocked?: boolean; // Flag indicating if the picklist is locked and should be read-only
}

// Progress indicator component for estimated time (non-batch processing)
const ProgressIndicator: React.FC<{ estimatedTime: number; teamCount: number }> = ({ estimatedTime, teamCount }) => {
  const [elapsedTime, setElapsedTime] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0);
  
  // Set up timer
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(prev => {
        const newTime = prev + 0.1;
        // Update progress
        setProgress(Math.min((newTime / estimatedTime) * 100, 99));
        return newTime;
      });
    }, 100); // Update every 100ms
    
    return () => clearInterval(timer);
  }, [estimatedTime]);
  
  return (
    <div className="flex flex-col items-center w-full max-w-lg mx-auto my-8 px-4">
      <div className="text-center mb-4 space-y-2">
        <h3 className="text-xl font-semibold text-blue-600">
          Generating Picklist for {teamCount} Teams
        </h3>
        <p className="text-gray-600">
          Estimated time: ~{Math.round(estimatedTime)} seconds
        </p>
        <p className="text-gray-600">
          Time elapsed: {elapsedTime.toFixed(1)} seconds
        </p>
      </div>
      
      <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
        <div 
          className="bg-blue-600 h-4 rounded-full transition-all duration-100 ease-out"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      
      <p className="text-sm text-gray-500">
        {progress < 30 ? 'Starting...' : 
         progress < 60 ? 'Processing team data...' : 
         progress < 90 ? 'Generating rankings...' : 
         'Finalizing picklist...'}
      </p>
    </div>
  );
};

// Batch progress component
const BatchProgressIndicator: React.FC<{ 
  batchInfo: BatchProcessing; 
  teamCount: number; 
  elapsedTime: number;
}> = ({ batchInfo, teamCount, elapsedTime }) => {
  return (
    <div className="flex flex-col items-center w-full max-w-lg mx-auto my-8 px-4">
      <div className="text-center mb-4 space-y-2">
        <h3 className="text-xl font-semibold text-blue-600">
          Generating Picklist in Batches
        </h3>
        <p className="text-gray-600">
          Processing teams in batches for higher quality rankings (updates every 5 seconds)
        </p>
        <p className="text-gray-600">
          Time elapsed: {elapsedTime.toFixed(1)} seconds
        </p>
      </div>
      
      <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
        <div 
          className="bg-blue-600 h-4 rounded-full transition-all duration-100 ease-out"
          style={{ width: `${batchInfo.progress_percentage}%` }}
        ></div>
      </div>
      
      <div className="flex justify-between w-full text-sm text-gray-600 mt-1 mb-3">
        <span>Processing batch {batchInfo.current_batch} of {batchInfo.total_batches}</span>
        <span>{batchInfo.progress_percentage}% complete</span>
      </div>
      
      <p className="text-sm text-gray-500">
        {batchInfo.progress_percentage === 0 ? 'Starting batch processing...' : 
         batchInfo.progress_percentage < 30 ? 'Processing first batches...' : 
         batchInfo.progress_percentage < 70 ? 'Ranking teams with reference teams...' : 
         batchInfo.progress_percentage < 95 ? 'Processing final batches...' : 
         'Combining results...'}
      </p>
    </div>
  );
};

// Modal for Missing Teams
const MissingTeamsModal: React.FC<{
  missingTeamCount: number;
  onRankMissingTeams: () => void;
  onSkip: () => void;
  isLoading: boolean;
}> = ({ missingTeamCount, onRankMissingTeams, onSkip, isLoading }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full">
        <h3 className="text-xl font-bold mb-4">Auto-Added Teams Detected</h3>
        <p className="mb-6">
          {missingTeamCount} teams have been automatically added with estimated scores. Would you like to get more accurate rankings for these teams?
        </p>
        <div className="flex justify-end space-x-3">
          <button
            onClick={onSkip}
            disabled={isLoading}
            className="px-4 py-2 bg-gray-300 text-gray-800 rounded hover:bg-gray-400 disabled:opacity-50"
          >
            Skip
          </button>
          <button
            onClick={onRankMissingTeams}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 flex items-center"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></div>
                Ranking...
              </>
            ) : (
              'Get Accurate Rankings'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

const PicklistGenerator: React.FC<PicklistGeneratorProps> = ({
  datasetPath,
  yourTeamNumber,
  pickPosition,
  priorities,
  strategyPrompt = "",
  excludeTeams = [],
  onPicklistGenerated,
  initialPicklist = [],
  onExcludeTeam,
  isLocked = false
}) => {
  const [picklist, setPicklist] = useState<Team[]>(initialPicklist);
  const [analysis, setAnalysis] = useState<PicklistAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [estimatedTime, setEstimatedTime] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [showAnalysis, setShowAnalysis] = useState<boolean>(false);
  const [useSemanticBatching, setUseSemanticBatching] = useState<boolean>(true);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  
  // Missing teams state
  const [missingTeamNumbers, setMissingTeamNumbers] = useState<number[]>([]);
  const [showMissingTeamsModal, setShowMissingTeamsModal] = useState<boolean>(false);
  const [isRankingMissingTeams, setIsRankingMissingTeams] = useState<boolean>(false);
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [teamsPerPage, setTeamsPerPage] = useState<number>(10);
  const [totalPages, setTotalPages] = useState<number>(1);
  
  // Batch processing state
  const [batchProcessingActive, setBatchProcessingActive] = useState<boolean>(false);
  const [batchProcessingInfo, setBatchProcessingInfo] = useState<BatchProcessing | null>(null);
  const [pollingCacheKey, setPollingCacheKey] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0);
  
  useEffect(() => {
    // Log when dependencies change
    console.log("PicklistGenerator dependencies changed:", { 
      pickPosition, 
      prioritiesCount: priorities.length,
      excludeTeamsCount: excludeTeams?.length
    });
    
    if (excludeTeams && excludeTeams.length > 0) {
      console.log("Teams to exclude:", excludeTeams);
    }
    
    // If we have an initial picklist, use it
    if (initialPicklist && initialPicklist.length > 0) {
      setPicklist(initialPicklist);
      // Calculate total pages
      setTotalPages(Math.ceil(initialPicklist.length / teamsPerPage));
    }
    
    /* 
     * IMPORTANT NOTE ON AUTOMATIC REGENERATION:
     * We deliberately removed the automatic generatePicklist() call that was here previously.
     * This prevents unwanted regeneration when navigating between pages (especially when
     * returning from Alliance Selection back to Picklist).
     * 
     * Rankings will ONLY be generated when the user explicitly clicks the "Generate Rankings" button.
     * This ensures the user experience is predictable and prevents unexpected API calls.
     * 
     * This change was made on 2025-05-09 to improve navigation between the
     * Picklist and Alliance Selection pages.
     */
    
    // Reset to page 1 when dependencies change
    setCurrentPage(1);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetPath, yourTeamNumber, pickPosition]); 
  
  // Add a separate effect to handle deep changes to priorities and excludeTeams
  useEffect(() => {
    // This effect will only run when priorities or excludeTeams change in a way that affects the actual data
    // This prevents the problem of JSON.stringify creating a new string on every render
    console.log("Priorities or exclusions changed significantly");
    
    // We've removed the automatic regeneration here too
    // The user must explicitly click "Generate Rankings" to trigger regeneration
    // This prevents unexpected regeneration when navigating between tabs
    
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    // Use the length as a proxy for deep changes
    priorities.length, 
    excludeTeams?.length,
    // We need datasetPath and yourTeamNumber for the check inside
    datasetPath,
    yourTeamNumber
  ]);
  
  // Update totalPages when teamsPerPage changes
  useEffect(() => {
    setTotalPages(Math.ceil(picklist.length / teamsPerPage));
  }, [picklist.length, teamsPerPage]);
  
  // Elapsed time counter for batch processing
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    
    if (batchProcessingActive) {
      // Start a timer that updates every 100ms
      timer = setInterval(() => {
        setElapsedTime(prev => prev + 0.1);
      }, 100);
    } else {
      // Reset elapsed time when batch processing stops
      setElapsedTime(0);
    }
    
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [batchProcessingActive]);
  
  // Polling effect for batch processing status
  useEffect(() => {
    let pollingInterval: NodeJS.Timeout | null = null;
    
    if (batchProcessingActive && pollingCacheKey) {
      // Poll every 5 seconds to reduce OPTIONS requests
      pollingInterval = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8000/api/picklist/generate/status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cache_key: pollingCacheKey })
          });
          
          if (!response.ok) {
            console.error('Error polling batch status:', response.status);
            return;
          }
          
          const data = await response.json();
          
          // Update batch processing info
          if (data.batch_processing) {
            setBatchProcessingInfo(data.batch_processing);
            
            // If processing is complete, stop polling and process the final result
            if (data.batch_processing.processing_complete) {
              setBatchProcessingActive(false);
              clearInterval(pollingInterval);
              
              // Process the picklist data if it's included with the completion response
              if (data.picklist) {
                console.log('Processing completed picklist data:', data);
                
                // Process the picklist data
                setPicklist(data.picklist);
                if (data.analysis) setAnalysis(data.analysis);
                
                // Reset batch processing state
                setBatchProcessingInfo(null);
                setPollingCacheKey(null);
                
                // Reset to page 1
                setCurrentPage(1);
                setTotalPages(Math.ceil(data.picklist.length / teamsPerPage));
                
                // Call the callback if provided
                if (onPicklistGenerated) {
                  onPicklistGenerated(data);
                }
              } else {
                // Fall back to fetching the result explicitly if picklist data isn't included
                fetchCompletedPicklist();
              }
            }
          }
        } catch (err) {
          console.error('Error polling batch status:', err);
        }
      }, 5000); // Increased from 2000ms to 5000ms to reduce OPTIONS requests
    }
    
    return () => {
      if (pollingInterval) clearInterval(pollingInterval);
    };
  }, [batchProcessingActive, pollingCacheKey]);
  
  // Function to fetch the completed picklist
  const fetchCompletedPicklist = async () => {
    if (!pollingCacheKey) return;
    
    try {
      // Fetch the completed picklist using the cache key
      const response = await fetch('http://localhost:8000/api/picklist/generate/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cache_key: pollingCacheKey })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch completed picklist');
      }
      
      const data = await response.json();
      
      // If we have a complete result with picklist data, process it
      if (data.status === 'success' && data.picklist) {
        // Process the picklist data
        setPicklist(data.picklist);
        if (data.analysis) setAnalysis(data.analysis);
        
        // Reset batch processing state
        setBatchProcessingActive(false);
        setBatchProcessingInfo(null);
        setPollingCacheKey(null);
        
        // Reset to page 1
        setCurrentPage(1);
        setTotalPages(Math.ceil(data.picklist.length / teamsPerPage));
        
        // Call the callback if provided
        if (onPicklistGenerated) {
          onPicklistGenerated(data);
        }
      }
    } catch (err) {
      console.error('Error fetching completed picklist:', err);
      setError('Failed to fetch the completed picklist');
      
      // Reset batch processing state
      setBatchProcessingActive(false);
      setBatchProcessingInfo(null);
      setPollingCacheKey(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  const generatePicklist = async () => {
    if (!datasetPath || !yourTeamNumber || !priorities.length) {
      setError('Missing required inputs for picklist generation');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Reset batch processing state
    setBatchProcessingActive(false);
    setBatchProcessingInfo(null);
    setPollingCacheKey(null);
    
    // Calculate estimated time - approximate from team count
    const teamsToRank = excludeTeams ? 75 - excludeTeams.length : 75;
    const estimatedSeconds = teamsToRank * 0.9; // ~0.9 seconds per team
    setEstimatedTime(estimatedSeconds);
    
    try {
      // Send the priorities as plain JSON objects without any methods or class structures
      const simplePriorities = [];
      for (const priority of priorities) {
        simplePriorities.push({
          id: priority.id,
          weight: priority.weight,
          reason: priority.reason || null
        });
      }
      
      // Create a request object with all primitive values
      // Ensure excludeTeams is always an array and log it
      const teamsToExclude = excludeTeams || [];
      console.log(`Excluding ${teamsToExclude.length} teams for ${pickPosition} pick:`, teamsToExclude);
      
      // Create request body with batching parameters
      const requestBody = JSON.stringify({
        unified_dataset_path: datasetPath,
        your_team_number: yourTeamNumber,
        pick_position: pickPosition,
        priorities: simplePriorities,
        strategy_prompt: strategyPrompt,  // Include strategy prompt if provided
        exclude_teams: teamsToExclude,
        use_batching: true,  // Enable batch processing by default
        batch_size: 20,
        reference_teams_count: 3,
        reference_selection: "top_middle_bottom",
        use_semantic_batching: useSemanticBatching  // Use the semantic batching state
      });
      
      console.log('Sending request:', requestBody);
      
      const response = await fetch('http://localhost:8000/api/picklist/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate picklist');
      }
      
      const data: PicklistResult = await response.json();
      
      // Check if this is a batched result
      if (data.status === 'success' && data.batched) {
        console.log('Batch processing initiated', data);
        
        // If we have a cache key, start polling for updates
        if (data.cache_key) {
          // Start batch processing monitoring
          setBatchProcessingActive(true);
          setPollingCacheKey(data.cache_key);
          
          // Set initial batch processing info if available
          if (data.batch_processing) {
            setBatchProcessingInfo(data.batch_processing);
          } else {
            // Default initial values
            setBatchProcessingInfo({
              total_batches: 0,
              current_batch: 0,
              progress_percentage: 0,
              processing_complete: false
            });
          }
          
          // Don't set picklist until batch processing is complete
          return;
        }
      }
      
      // For non-batched results or immediate completion, process normally
      if (data.status === 'success') {
        // Reset batch processing state
        setBatchProcessingActive(false);
        setBatchProcessingInfo(null);
        setPollingCacheKey(null);
        
        setPicklist(data.picklist);
        setAnalysis(data.analysis);
        
        // Reset estimated time
        setEstimatedTime(0);
        setTotalPages(Math.ceil(data.picklist.length / teamsPerPage));
        setCurrentPage(1); // Reset to first page
        
        // Check for missing teams
        if (data.missing_team_numbers && data.missing_team_numbers.length > 0) {
          // Count how many teams are actually auto-added fallbacks in the picklist
          const autoAddedTeamsCount = data.picklist.filter(team => team.is_fallback).length;
          
          if (autoAddedTeamsCount > 0) {
            setMissingTeamNumbers(data.missing_team_numbers);
            setShowMissingTeamsModal(true);
          } else {
            // If no auto-added teams in the picklist, don't show the modal
            setMissingTeamNumbers([]);
          }
        } else {
          setMissingTeamNumbers([]);
        }
        
        // Call the callback if provided
        if (onPicklistGenerated) {
          onPicklistGenerated(data);
        }
      } else {
        setError(data.message || 'Error generating picklist');
      }
    } catch (err: any) {
      setError(err.message || 'Error connecting to server');
      console.error('Error generating picklist:', err);
      
      // Reset batch processing state on error
      setBatchProcessingActive(false);
      setBatchProcessingInfo(null);
      setPollingCacheKey(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  const updatePicklist = async () => {
    if (!datasetPath || !picklist.length) {
      setError('No picklist data to update');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Calculate estimated time - approximate from team count
    const teamsToRank = excludeTeams ? 75 - excludeTeams.length : 75;
    const estimatedSeconds = teamsToRank * 0.9; // ~0.9 seconds per team
    setEstimatedTime(estimatedSeconds);
    
    try {
      // Convert picklist to user rankings format
      const userRankings = picklist.map((team, index) => ({
        team_number: team.team_number,
        position: index + 1,
        nickname: team.nickname
      }));
      
      const response = await fetch('http://localhost:8000/api/picklist/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          unified_dataset_path: datasetPath,
          original_picklist: picklist,
          user_rankings: userRankings
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to update picklist');
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setPicklist(data.picklist);
        setIsEditing(false);
      } else {
        setError(data.message || 'Error updating picklist');
      }
    } catch (err: any) {
      setError(err.message || 'Error connecting to server');
      console.error('Error updating picklist:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Function to rank missing teams
  const rankMissingTeams = async () => {
    if (!datasetPath || !missingTeamNumbers.length) {
      setError('No missing teams to rank');
      return;
    }
    
    setIsRankingMissingTeams(true);
    setError(null);
    
    try {
      // Convert priorities to the format expected by the backend
      const simplePriorities = [];
      for (const priority of priorities) {
        simplePriorities.push({
          id: priority.id,
          weight: priority.weight,
          reason: priority.reason || null
        });
      }
      
      // Make the API call to rank missing teams
      const response = await fetch('http://localhost:8000/api/picklist/rank-missing-teams', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          unified_dataset_path: datasetPath,
          missing_team_numbers: missingTeamNumbers,
          ranked_teams: picklist,
          your_team_number: yourTeamNumber,
          pick_position: pickPosition,
          priorities: simplePriorities
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to rank missing teams');
      }
      
      const data: MissingTeamsResult = await response.json();
      
      if (data.status === 'success' && data.missing_team_rankings) {
        // Filter out any fallback teams that were re-ranked
        const rerankedTeamNumbers = data.missing_team_rankings.map(team => team.team_number);
        const filteredPicklist = picklist.filter(team => 
          !team.is_fallback || !rerankedTeamNumbers.includes(team.team_number)
        );
        
        // Merge properly ranked missing teams with filtered picklist
        const updatedPicklist = [...filteredPicklist, ...data.missing_team_rankings];
        
        // Sort by score (highest to lowest)
        updatedPicklist.sort((a, b) => b.score - a.score);
        
        // Update state
        setPicklist(updatedPicklist);
        setTotalPages(Math.ceil(updatedPicklist.length / teamsPerPage));
        
        // Show success message
        setSuccessMessage(`Successfully replaced ${data.missing_team_rankings.length} auto-added teams with proper rankings!`);
        setTimeout(() => setSuccessMessage(null), 3000);
        
        // Call the callback if provided
        if (onPicklistGenerated) {
          onPicklistGenerated({
            status: 'success',
            picklist: updatedPicklist,
            analysis: analysis as PicklistAnalysis
          });
        }
      } else {
        setError(data.message || 'Error ranking missing teams');
      }
    } catch (err: any) {
      setError(err.message || 'Error connecting to server');
      console.error('Error ranking missing teams:', err);
    } finally {
      setIsRankingMissingTeams(false);
      setShowMissingTeamsModal(false);
    }
  };
  
  // Handle skipping the missing teams ranking
  const handleSkipMissingTeams = () => {
    setShowMissingTeamsModal(false);
    setMissingTeamNumbers([]);
    
    // When skipping, ensure the picklist is properly sorted by score
    const sortedPicklist = [...picklist].sort((a, b) => b.score - a.score);
    setPicklist(sortedPicklist);
  };
  
  // Handle team position change
  const handlePositionChange = (teamIndex: number, newPosition: number) => {
    // Validate the new position
    if (newPosition < 1 || newPosition > picklist.length) {
      setError(`Position must be between 1 and ${picklist.length}`);
      setTimeout(() => setError(null), 3000);
      return;
    }
    
    // Convert from 1-based UI position to 0-based array index
    const newIndex = newPosition - 1;
    
    // Create a copy of the current picklist
    const newPicklist = [...picklist];
    
    // Remove the team from its current position
    const [teamToMove] = newPicklist.splice(teamIndex, 1);
    
    // Insert the team at the new position
    newPicklist.splice(newIndex, 0, teamToMove);
    
    // Update the picklist
    setPicklist(newPicklist);
    
    // Recalculate total pages
    setTotalPages(Math.ceil(newPicklist.length / teamsPerPage));
    
    // Show feedback that rankings changed
    setSuccessMessage('Team position updated');
    setTimeout(() => setSuccessMessage(null), 2000);
  };
  
  // Decide which loading/progress indicator to show
  if ((isLoading && !picklist.length) || batchProcessingActive) {
    // If batch processing is active, show the batch progress indicator
    if (batchProcessingActive && batchProcessingInfo) {
      return (
        <BatchProgressIndicator 
          batchInfo={batchProcessingInfo} 
          teamCount={datasetPath ? 75 : 0} 
          elapsedTime={elapsedTime} 
        />
      );
    }
    
    // Show estimated time progress indicator for non-batch processing
    if (estimatedTime > 0) {
      return <ProgressIndicator estimatedTime={estimatedTime} teamCount={datasetPath ? 75 : 0} />;
    }
    
    // Fall back to simple spinner if no time estimate
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-blue-600">Generating picklist...</span>
      </div>
    );
  }
  
  // Don't show the picklist if batch processing is active
  if (batchProcessingActive) {
    return null;
  }
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Missing Teams Modal */}
      {showMissingTeamsModal && (
        <MissingTeamsModal
          missingTeamCount={missingTeamNumbers.length}
          onRankMissingTeams={rankMissingTeams}
          onSkip={handleSkipMissingTeams}
          isLoading={isRankingMissingTeams}
        />
      )}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">
          {pickPosition.charAt(0).toUpperCase() + pickPosition.slice(1)} Pick Rankings
        </h2>
        <div className="flex space-x-2">
          {isEditing ? (
            <>
              <button
                onClick={updatePicklist}
                disabled={isLoading}
                className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-green-300"
              >
                Save Changes
              </button>
              <button
                onClick={() => setIsEditing(false)}
                className="px-3 py-1 bg-gray-400 text-white rounded hover:bg-gray-500"
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              {!isLocked && (
                <button
                  onClick={() => setIsEditing(true)}
                  className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Edit Rankings
                </button>
              )}
              <button
                onClick={() => setShowAnalysis(!showAnalysis)}
                className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                {showAnalysis ? 'Hide Analysis' : 'Show Analysis'}
              </button>
              {!isLocked && (
                <div className="flex items-center space-x-2">
                  <div className="flex items-center mr-2 px-2 py-1 border border-blue-300 rounded bg-blue-50">
                    <input
                      type="checkbox"
                      id="semanticBatching"
                      checked={useSemanticBatching}
                      onChange={(e) => setUseSemanticBatching(e.target.checked)}
                      className="mr-1 h-4 w-4 text-blue-600 rounded"
                    />
                    <label htmlFor="semanticBatching" className="text-sm text-blue-700 font-semibold">
                      Smart Batching
                    </label>
                    <span className="ml-1 text-blue-500 text-xs cursor-help" title="Groups similar teams for more consistent rankings">
                      ℹ️
                    </span>
                  </div>
                  <button
                    onClick={generatePicklist}
                    disabled={isLoading}
                    className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-green-300"
                  >
                    {isLoading ? 'Regenerating...' : 'Regenerate Picklist'}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      
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
      
      {showAnalysis && analysis && (
        <div className="mb-6 bg-purple-50 p-4 rounded-lg border border-purple-200">
          <h3 className="font-bold text-purple-800 mb-2">GPT Analysis</h3>
          
          <div className="space-y-3">
            <div>
              <h4 className="font-semibold text-purple-700">Draft Reasoning:</h4>
              <p className="text-sm text-gray-700">{analysis.draft_reasoning}</p>
            </div>
            
            <div>
              <h4 className="font-semibold text-purple-700">Critical Evaluation:</h4>
              <p className="text-sm text-gray-700">{analysis.evaluation}</p>
            </div>
            
            <div>
              <h4 className="font-semibold text-purple-700">Final Recommendations:</h4>
              <p className="text-sm text-gray-700">{analysis.final_recommendations}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Pagination Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-center mb-4 py-2 border-b border-t">
        <div className="mb-2 sm:mb-0">
          <span className="mr-2">Teams per page:</span>
          <select
            value={teamsPerPage}
            onChange={(e) => {
              const newTeamsPerPage = parseInt(e.target.value);
              setTeamsPerPage(newTeamsPerPage);
              // Adjust current page to keep showing the first team currently visible
              const firstTeamOnCurrentPage = (currentPage - 1) * teamsPerPage + 1;
              const newPage = Math.ceil(firstTeamOnCurrentPage / newTeamsPerPage);
              setCurrentPage(Math.max(1, Math.min(newPage, Math.ceil(picklist.length / newTeamsPerPage))));
            }}
            className="border rounded p-1"
          >
            <option value="5">5</option>
            <option value="10">10</option>
            <option value="15">15</option>
            <option value="25">25</option>
            <option value="50">50</option>
          </select>
        </div>
        
        <div className="flex items-center">
          <button
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
            className={`px-2 py-1 rounded ${currentPage === 1 ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="First page"
          >
            &laquo;
          </button>
          <button
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className={`px-3 py-1 rounded ${currentPage === 1 ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="Previous page"
          >
            &lsaquo;
          </button>
          
          <span className="mx-2">
            Page {currentPage} of {totalPages || 1}
          </span>
          
          <button
            onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
            disabled={currentPage >= totalPages}
            className={`px-3 py-1 rounded ${currentPage >= totalPages ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="Next page"
          >
            &rsaquo;
          </button>
          <button
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage >= totalPages}
            className={`px-2 py-1 rounded ${currentPage >= totalPages ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="Last page"
          >
            &raquo;
          </button>
        </div>
        
        <div className="mt-2 sm:mt-0 text-sm text-gray-500">
          {picklist.length > 0 ? (
            <span>
              Showing {Math.min((currentPage - 1) * teamsPerPage + 1, picklist.length)}-
              {Math.min(currentPage * teamsPerPage, picklist.length)} of {picklist.length} teams
            </span>
          ) : (
            <span>No teams to display</span>
          )}
        </div>
      </div>
      
      <div className="overflow-hidden">
        {isEditing ? (
          <div className="space-y-3">
            <p className="text-sm text-blue-600 italic mb-2">
              Edit team positions by changing their rank numbers, then click "Save Changes" when done.
            </p>
            
            {/* Get current page of teams */}
            {picklist
              .slice((currentPage - 1) * teamsPerPage, currentPage * teamsPerPage)
              .map((team, pageIndex) => {
                // Get absolute index in full list
                const index = (currentPage - 1) * teamsPerPage + pageIndex;
                
                return (
                  <div 
                    key={team.team_number} 
                    className="p-3 bg-white rounded border border-gray-300 shadow-sm flex items-center hover:bg-blue-50 transition-colors duration-150"
                  >
                    <div className="mr-3 flex items-center">
                      <input
                        type="number"
                        min="1"
                        max={picklist.length}
                        value={index + 1}
                        onChange={(e) => handlePositionChange(index, parseInt(e.target.value) || 1)}
                        className="w-12 p-1 border border-gray-300 rounded text-center font-bold text-blue-600"
                      />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">
                        Team {team.team_number}: {team.nickname}
                        {team.is_fallback && (
                          <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-100 text-yellow-800 rounded-full" title="This team was automatically added to complete the picklist">
                            Auto-added
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-600">Score: {team.score.toFixed(2)}</div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {onExcludeTeam && !isLocked && (
                        <button
                          onClick={() => onExcludeTeam(team.team_number)}
                          className="px-2 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700"
                          title="Exclude team from all picklists"
                        >
                          Exclude
                        </button>
                      )}
                      <div className="text-gray-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
                        </svg>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        ) : (
          <div className="space-y-2">
            {/* Get current page of teams */}
            {picklist
              .slice((currentPage - 1) * teamsPerPage, currentPage * teamsPerPage)
              .map((team, pageIndex) => {
                // Get absolute index in full list
                const index = (currentPage - 1) * teamsPerPage + pageIndex;
                
                return (
                  <div key={team.team_number} className="p-3 bg-white rounded border flex hover:bg-gray-50">
                    <div className="mr-3 text-lg font-bold text-gray-500">{index + 1}</div>
                    <div className="flex-1">
                      <div className="font-medium">
                        Team {team.team_number}: {team.nickname}
                        {team.is_fallback && (
                          <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-100 text-yellow-800 rounded-full" title="This team was automatically added to complete the picklist">
                            Auto-added
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-600">Score: {team.score.toFixed(2)}</div>
                      <div className={`text-sm mt-1 ${team.is_fallback ? 'italic text-yellow-700' : ''}`}>
                        {team.reasoning}
                      </div>
                      {team.is_fallback && (
                        <div className="text-xs mt-1 text-red-500">
                          Note: This team was added automatically because it was missing from the GPT response.
                        </div>
                      )}
                    </div>
                    {onExcludeTeam && !isLocked && (
                      <div className="ml-2 flex items-center">
                        <button
                          onClick={() => onExcludeTeam(team.team_number)}
                          className="px-2 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700"
                          title="Exclude team from all picklists"
                        >
                          Exclude
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        )}
      </div>
      
      {/* Bottom Pagination for convenience with longer lists */}
      {totalPages > 1 && (
        <div className="mt-4 flex justify-center">
          <button
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
            className={`px-2 py-1 rounded ${currentPage === 1 ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="First page"
          >
            &laquo;
          </button>
          <button
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className={`px-3 py-1 rounded ${currentPage === 1 ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="Previous page"
          >
            &lsaquo;
          </button>
          
          {/* Page number buttons */}
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            // Show pages around current page
            let pageNum;
            if (totalPages <= 5) {
              pageNum = i + 1;
            } else if (currentPage <= 3) {
              pageNum = i + 1;
            } else if (currentPage >= totalPages - 2) {
              pageNum = totalPages - 4 + i;
            } else {
              pageNum = currentPage - 2 + i;
            }
            
            return (
              <button
                key={pageNum}
                onClick={() => setCurrentPage(pageNum)}
                className={`px-3 py-1 mx-1 rounded ${currentPage === pageNum ? 'bg-blue-600 text-white' : 'text-blue-600 hover:bg-blue-50'}`}
              >
                {pageNum}
              </button>
            );
          })}
          
          <button
            onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
            disabled={currentPage >= totalPages}
            className={`px-3 py-1 rounded ${currentPage >= totalPages ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="Next page"
          >
            &rsaquo;
          </button>
          <button
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage >= totalPages}
            className={`px-2 py-1 rounded ${currentPage >= totalPages ? 'text-gray-400' : 'text-blue-600 hover:bg-blue-50'}`}
            title="Last page"
          >
            &raquo;
          </button>
        </div>
      )}
    </div>
  );
};

export default PicklistGenerator;