import { useState, useEffect } from 'react';
import { ListChecks, Play, RefreshCw, AlertCircle } from 'lucide-react';
import { getScenarios, predictScenario } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PredictionResult from '../components/prediction/PredictionResult';

function ScenarioSelect() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingScenarios, setLoadingScenarios] = useState(true);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Load scenarios on mount
  useEffect(() => {
    loadScenarios();
  }, []);

  const loadScenarios = async () => {
    setLoadingScenarios(true);
    setError(null);
    
    try {
      const response = await getScenarios();
      setScenarios(response.scenarios || []);
    } catch (err) {
      setError('Failed to load scenarios. Please check if the server is running.');
      console.error('Load scenarios error:', err);
    } finally {
      setLoadingScenarios(false);
    }
  };

  const handlePredict = async () => {
    if (!selectedScenario) {
      setError('Please select a scenario first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictScenario(selectedScenario.id);
      setResult(response);
      
      // Scroll to result
      setTimeout(() => {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedScenario(null);
    setResult(null);
    setError(null);
  };

  // Quality badge colors
  const getQualityColor = (quality) => {
    const colors = {
      Good: 'bg-success-50 text-success-700 border-success-200',
      Moderate: 'bg-warning-50 text-warning-700 border-warning-200',
      Poor: 'bg-danger-50 text-danger-700 border-danger-200',
    };
    return colors[quality] || 'bg-gray-50 text-gray-700 border-gray-200';
  };

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-3 bg-success-100 rounded-lg">
            <ListChecks className="w-8 h-8 text-success-600" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Scenario Selection</h1>
            <p className="text-gray-600">Choose from realistic Vietnamese network scenarios</p>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Scenario List */}
        <div className="space-y-4 animate-slide-up">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900">
              Available Scenarios ({scenarios.length})
            </h2>
            <button
              onClick={loadScenarios}
              disabled={loadingScenarios}
              className="text-sm text-primary-600 hover:text-primary-700 flex items-center space-x-1"
            >
              <RefreshCw className={`w-4 h-4 ${loadingScenarios ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>

          {loadingScenarios ? (
            <div className="card">
              <LoadingSpinner text="Loading scenarios..." />
            </div>
          ) : error && scenarios.length === 0 ? (
            <div className="card bg-danger-50 border-danger-200">
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-danger-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm text-danger-700 mb-2">{error}</p>
                  <button onClick={loadScenarios} className="text-sm text-danger-600 hover:text-danger-700 underline">
                    Try again
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {scenarios.map((scenario) => (
                <button
                  key={scenario.id}
                  onClick={() => setSelectedScenario(scenario)}
                  className={`w-full text-left card transition-all duration-200 hover:shadow-lg ${
                    selectedScenario?.id === scenario.id
                      ? 'ring-2 ring-primary-500 border-primary-300 bg-primary-50'
                      : 'hover:border-primary-200'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h3 className="font-bold text-gray-900 mb-1">
                        {scenario.icon || '📍'} {scenario.name}
                      </h3>
                      <p className="text-sm text-gray-600">{scenario.description}</p>
                    </div>
                    {selectedScenario?.id === scenario.id && (
                      <div className="ml-2 w-6 h-6 rounded-full bg-primary-600 flex items-center justify-center flex-shrink-0">
                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className={`text-xs px-2 py-1 rounded-full border ${getQualityColor(scenario.expected_quality)}`}>
                      Expected: {scenario.expected_quality}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Action Buttons */}
          {selectedScenario && (
            <div className="card bg-primary-50 border-primary-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Selected Scenario:</p>
                  <p className="font-semibold text-gray-900">{selectedScenario.name}</p>
                </div>
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="btn-primary flex items-center space-x-2"
                >
                  {loading ? (
                    <>
                      <LoadingSpinner size="sm" text="" />
                      <span>Predicting...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Predict</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && !loadingScenarios && scenarios.length > 0 && (
            <div className="card bg-danger-50 border-danger-200">
              <div className="flex items-start space-x-2">
                <AlertCircle className="w-5 h-5 text-danger-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-danger-700">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Result Section */}
        <div id="result-section" className="lg:sticky lg:top-8 lg:self-start">
          {loading && (
            <div className="card animate-fade-in">
              <LoadingSpinner text="Running scenario prediction..." />
            </div>
          )}
          
          {result && !loading && (
            <PredictionResult result={result} />
          )}
          
          {!result && !loading && (
            <div className="card bg-gray-50 text-center">
              <div className="py-8">
                <ListChecks className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">
                  Select a scenario and click "Predict" to see results
                </p>
                {selectedScenario && (
                  <button
                    onClick={handlePredict}
                    className="btn-primary mx-auto mt-4"
                  >
                    Predict with Selected Scenario
                  </button>
                )}
              </div>
            </div>
          )}
          
          {result && (
            <button
              onClick={handleReset}
              className="btn-secondary w-full mt-4 flex items-center justify-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Try Another Scenario</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default ScenarioSelect;