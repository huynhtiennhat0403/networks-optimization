import { useState } from 'react';
import { Zap, Send, RefreshCw, Info } from 'lucide-react';
import { predictSimple } from '../services/api';
import { 
  USER_ACTIVITIES, 
  DEVICE_TYPES, 
  LOCATIONS, 
  CONNECTION_TYPES, 
  DEFAULT_VALUES,
  INPUT_RANGES 
} from '../services/constants';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PredictionResult from '../components/prediction/PredictionResult';

function SmartInput() {
  const [formData, setFormData] = useState(DEFAULT_VALUES);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError(null);
  };

  const handleReset = () => {
    setFormData(DEFAULT_VALUES);
    setResult(null);
    setError(null);
  };

  const validateInput = () => {
    const { throughput, latency, signal_strength } = formData;
    
    if (throughput < INPUT_RANGES.throughput.min || throughput > INPUT_RANGES.throughput.max) {
      return `Throughput must be between ${INPUT_RANGES.throughput.min}-${INPUT_RANGES.throughput.max} Mbps`;
    }
    if (latency < INPUT_RANGES.latency.min || latency > INPUT_RANGES.latency.max) {
      return `Latency must be between ${INPUT_RANGES.latency.min}-${INPUT_RANGES.latency.max} ms`;
    }
    if (signal_strength < INPUT_RANGES.signal_strength.min || signal_strength > INPUT_RANGES.signal_strength.max) {
      return `Signal strength must be between ${INPUT_RANGES.signal_strength.min} to ${INPUT_RANGES.signal_strength.max} dBm`;
    }
    
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const validationError = validateInput();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictSimple(formData);
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

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-3 bg-primary-100 rounded-lg">
            <Zap className="w-8 h-8 text-primary-600" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Smart Input Mode</h1>
            <p className="text-gray-600">Enter 5 basic metrics, AI estimates the rest</p>
          </div>
        </div>
        
        <div className="flex items-start space-x-2 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-blue-800">
            Provide basic network metrics and optional context. Our AI will estimate 10+ additional 
            parameters using smart heuristics to predict network quality.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="card animate-slide-up">
          <h2 className="text-xl font-bold text-gray-900 mb-6">Network Metrics</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Required Metrics */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Required Metrics (3)
              </h3>
              
              {/* Throughput */}
              <div>
                <label className="label">
                  Throughput <span className="text-danger-500">*</span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.throughput}
                    onChange={(e) => handleChange('throughput', parseFloat(e.target.value))}
                    className="input-field"
                    step={INPUT_RANGES.throughput.step}
                    min={INPUT_RANGES.throughput.min}
                    max={INPUT_RANGES.throughput.max}
                    required
                  />
                  <span className="text-gray-600 text-sm whitespace-nowrap">
                    {INPUT_RANGES.throughput.unit}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Range: {INPUT_RANGES.throughput.min}-{INPUT_RANGES.throughput.max} Mbps
                </p>
              </div>

              {/* Latency */}
              <div>
                <label className="label">
                  Latency <span className="text-danger-500">*</span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.latency}
                    onChange={(e) => handleChange('latency', parseFloat(e.target.value))}
                    className="input-field"
                    step={INPUT_RANGES.latency.step}
                    min={INPUT_RANGES.latency.min}
                    max={INPUT_RANGES.latency.max}
                    required
                  />
                  <span className="text-gray-600 text-sm whitespace-nowrap">
                    {INPUT_RANGES.latency.unit}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Range: {INPUT_RANGES.latency.min}-{INPUT_RANGES.latency.max} ms
                </p>
              </div>

              {/* Signal Strength */}
              <div>
                <label className="label">
                  Signal Strength <span className="text-danger-500">*</span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.signal_strength}
                    onChange={(e) => handleChange('signal_strength', parseInt(e.target.value))}
                    className="input-field"
                    step={INPUT_RANGES.signal_strength.step}
                    min={INPUT_RANGES.signal_strength.min}
                    max={INPUT_RANGES.signal_strength.max}
                    required
                  />
                  <span className="text-gray-600 text-sm whitespace-nowrap">
                    {INPUT_RANGES.signal_strength.unit}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Range: {INPUT_RANGES.signal_strength.min} to {INPUT_RANGES.signal_strength.max} dBm
                </p>
              </div>
            </div>

            {/* Optional Context */}
            <div className="pt-6 border-t border-gray-200 space-y-4">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Optional Context (2)
              </h3>

              {/* User Activity */}
              <div>
                <label className="label">User Activity</label>
                <select
                  value={formData.user_activity}
                  onChange={(e) => handleChange('user_activity', e.target.value)}
                  className="input-field"
                >
                  {USER_ACTIVITIES.map(activity => (
                    <option key={activity.value} value={activity.value}>
                      {activity.icon} {activity.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Device Type */}
              <div>
                <label className="label">Device Type</label>
                <select
                  value={formData.device_type}
                  onChange={(e) => handleChange('device_type', e.target.value)}
                  className="input-field"
                >
                  {DEVICE_TYPES.map(device => (
                    <option key={device.value} value={device.value}>
                      {device.icon} {device.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Location */}
              <div>
                <label className="label">Location</label>
                <select
                  value={formData.location}
                  onChange={(e) => handleChange('location', e.target.value)}
                  className="input-field"
                >
                  {LOCATIONS.map(loc => (
                    <option key={loc.value} value={loc.value}>
                      {loc.icon} {loc.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Connection Type */}
              <div>
                <label className="label">Connection Type</label>
                <select
                  value={formData.connection_type}
                  onChange={(e) => handleChange('connection_type', e.target.value)}
                  className="input-field"
                >
                  {CONNECTION_TYPES.map(conn => (
                    <option key={conn.value} value={conn.value}>
                      {conn.icon} {conn.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-danger-50 border border-danger-200 rounded-lg">
                <p className="text-sm text-danger-700">{error}</p>
              </div>
            )}

            {/* Buttons */}
            <div className="flex space-x-3">
              <button
                type="submit"
                disabled={loading}
                className="btn-primary flex-1 flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <LoadingSpinner size="sm" text="" />
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Predict Quality</span>
                  </>
                )}
              </button>
              
              <button
                type="button"
                onClick={handleReset}
                className="btn-secondary flex items-center space-x-2"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Reset</span>
              </button>
            </div>
          </form>
        </div>

        {/* Result Section */}
        <div id="result-section" className="lg:sticky lg:top-8 lg:self-start">
          {loading && (
            <div className="card animate-fade-in">
              <LoadingSpinner text="Analyzing network parameters..." />
            </div>
          )}
          
          {result && !loading && (
            <PredictionResult 
              result={result} 
              estimatedParams={result.metadata?.estimated_features_dict}
            />
          )}
          
          {!result && !loading && (
            <div className="card bg-gray-50 text-center">
              <div className="py-8">
                <Zap className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">
                  Fill in the form and click "Predict Quality" to see results
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default SmartInput;