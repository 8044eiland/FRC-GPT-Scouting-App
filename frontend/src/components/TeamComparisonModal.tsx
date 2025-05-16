// TeamComparisonModal.tsx
import React, { useState, useEffect, useRef } from 'react';

interface Team {
  team_number: number;
  nickname: string;
  stats?: Record<string, number>;
  score?: number;
  metrics_contribution?: Array<{
    id: string;
    value: number;
    weighted_value: number;
    metrics_used?: string[];
  }>;
  match_count?: number;
}

interface MetricsComparison {
  [key: string]: {
    team1_value: number;
    team2_value: number;
    difference: number;
    better_team: number | null;
    significance: 'high' | 'medium' | 'low';
  };
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface MetricWeight {
  id: string;
  weight: number;
  reason?: string;
}

interface TeamComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  team1: Team | null;
  team2: Team | null;
  datasetPath: string;
  onSwapTeams: () => void;
  priorities?: MetricWeight[];
}

const TeamComparisonModal: React.FC<TeamComparisonModalProps> = ({
  isOpen,
  onClose,
  team1,
  team2,
  datasetPath,
  onSwapTeams,
  priorities = []
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [metricsComparison, setMetricsComparison] = useState<MetricsComparison | null>(null);
  const [analysisText, setAnalysisText] = useState<string>('');
  const [recommendation, setRecommendation] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  
  // References for the UI
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  
  // Load comparison data when modal opens
  useEffect(() => {
    if (isOpen && team1 && team2) {
      loadComparisonData();
    }
  }, [isOpen, team1, team2]);
  
  // Scroll to bottom of chat when history updates
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);
  
  // Function to load comparison data from API
  const loadComparisonData = async () => {
    if (!team1 || !team2) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/picklist/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          unified_dataset_path: datasetPath,
          team_numbers: [team1.team_number, team2.team_number],
          chat_history: [],
          specific_question: null,
          priorities: priorities
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to load comparison data');
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setMetricsComparison(data.comparison_data.metrics_comparison);
        setAnalysisText(data.comparison_data.qualitative_analysis);
        setRecommendation(data.comparison_data.recommendation);
        
        // Add the initial analysis to chat history
        setChatHistory([
          { role: 'assistant', content: data.comparison_data.chat_response }
        ]);
      } else {
        setError(data.message || 'Error loading comparison data');
      }
    } catch (err) {
      console.error('Error comparing teams:', err);
      setError('Error loading comparison data');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle sending a new chat message
  const handleSendMessage = async () => {
    if (!chatInput.trim() || !team1 || !team2) return;
    
    const userMessage = chatInput.trim();
    setChatInput('');
    
    // Add user message to chat history
    setChatHistory(prevChat => [...prevChat, { role: 'user', content: userMessage }]);
    
    setIsChatLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/picklist/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          unified_dataset_path: datasetPath,
          team_numbers: [team1.team_number, team2.team_number],
          chat_history: chatHistory,
          specific_question: userMessage,
          priorities: priorities
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to send message');
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        // Add assistant response to chat history
        setChatHistory(prevChat => [
          ...prevChat, 
          { role: 'assistant', content: data.comparison_data.chat_response }
        ]);
      } else {
        setError(data.message || 'Error sending message');
        // Add error message to chat history
        setChatHistory(prevChat => [
          ...prevChat, 
          { role: 'assistant', content: `Error: ${data.message || 'Failed to get response'}` }
        ]);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Error sending message');
      // Add error message to chat history
      setChatHistory(prevChat => [
        ...prevChat, 
        { role: 'assistant', content: 'Error: Failed to get response. Please try again.' }
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };
  
  // Format metric names for display
  const formatMetricName = (name: string): string => {
    // Remove prefixes like "statbotics_" or "superscout_"
    let displayName = name.replace(/^(statbotics_|superscout_)/, '');
    
    // Replace underscores with spaces and capitalize words
    displayName = displayName
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
      
    return displayName;
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b flex justify-between items-center">
          <h2 className="text-xl font-bold">
            Team Comparison: {team1?.team_number} vs {team2?.team_number}
          </h2>
          <div className="flex space-x-2">
            <button
              onClick={onSwapTeams}
              className="px-3 py-1 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center"
              title="Swap teams in picklist rankings and close"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
              Swap in Picklist
            </button>
            <button 
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-auto flex flex-col md:flex-row">
          {/* Left side - Metrics comparison */}
          <div className="w-full md:w-1/2 p-6 overflow-auto border-r">
            <h3 className="text-lg font-semibold mb-4">Key Metrics Comparison</h3>
            
            {isLoading ? (
              <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
              </div>
            ) : error ? (
              <div className="bg-red-100 text-red-700 p-4 rounded">
                {error}
              </div>
            ) : metricsComparison ? (
              <div className="space-y-4">
                {/* Team headers */}
                <div className="grid grid-cols-3 gap-4 font-semibold border-b pb-2">
                  <div>Metric</div>
                  <div className="text-center">{team1?.nickname || `Team ${team1?.team_number}`}</div>
                  <div className="text-center">{team2?.nickname || `Team ${team2?.team_number}`}</div>
                </div>
                
                {/* Metrics rows - sort by significance */}
                {Object.entries(metricsComparison)
                  .sort((a, b) => {
                    // First sort by significance
                    const significanceOrder = { high: 0, medium: 1, low: 2 };
                    const sigA = significanceOrder[a[1].significance];
                    const sigB = significanceOrder[b[1].significance];
                    
                    if (sigA !== sigB) return sigA - sigB;
                    
                    // Then by absolute difference
                    return Math.abs(b[1].difference) - Math.abs(a[1].difference);
                  })
                  .map(([metric, data]) => (
                    <div 
                      key={metric}
                      className={`grid grid-cols-3 gap-4 py-2 border-b ${
                        data.significance === 'high' 
                          ? 'bg-blue-50' 
                          : data.significance === 'medium'
                            ? 'bg-gray-50'
                            : ''
                      }`}
                    >
                      <div className="font-medium">{formatMetricName(metric)}</div>
                      <div 
                        className={`text-center ${
                          data.better_team === team1?.team_number ? 'font-bold text-green-600' : ''
                        }`}
                      >
                        {data.team1_value.toFixed(2)}
                      </div>
                      <div 
                        className={`text-center ${
                          data.better_team === team2?.team_number ? 'font-bold text-green-600' : ''
                        }`}
                      >
                        {data.team2_value.toFixed(2)}
                      </div>
                    </div>
                  ))
                }
                
                {/* Recommendation */}
                {recommendation && (
                  <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded">
                    <h4 className="text-md font-semibold text-blue-800 mb-2">Recommendation</h4>
                    <p className="text-blue-700">{recommendation}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-gray-500 italic">
                Select two teams to compare their metrics.
              </div>
            )}
          </div>
          
          {/* Right side - Chat interface */}
          <div className="w-full md:w-1/2 flex flex-col h-[600px]">
            <div className="flex-1 p-6 overflow-auto">
              <h3 className="text-lg font-semibold mb-4">Analysis & Discussion</h3>
              
              {/* Chat messages */}
              <div className="space-y-4">
                {chatHistory.map((message, index) => (
                  <div 
                    key={index}
                    className={`p-3 rounded-lg ${
                      message.role === 'assistant' 
                        ? 'bg-blue-100 mr-8' 
                        : 'bg-gray-100 ml-8'
                    }`}
                  >
                    <div className="text-xs font-medium mb-1">
                      {message.role === 'assistant' ? 'AI Analysis' : 'You'}
                    </div>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>
                ))}
                
                {isChatLoading && (
                  <div className="p-3 rounded-lg bg-blue-100 mr-8">
                    <div className="flex items-center">
                      <div className="text-xs font-medium mr-2">AI Analysis</div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Invisible element to scroll to */}
                <div ref={chatEndRef} />
              </div>
            </div>
            
            {/* Chat input */}
            <div className="p-4 border-t">
              <div className="flex items-center">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask a question about these teams..."
                  className="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isChatLoading}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isChatLoading || !chatInput.trim()}
                  className="bg-blue-600 text-white p-2 rounded-r-lg hover:bg-blue-700 disabled:bg-blue-300"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
              <div className="mt-2 text-xs text-gray-500">
                Ask specific questions like "Which team has better defense?" or "Compare their autonomous capabilities"
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamComparisonModal;