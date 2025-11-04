import { CheckCircle2, AlertTriangle, XCircle, TrendingUp, Activity, Info } from 'lucide-react';
import { QUALITY_COLORS, PARAMETER_DISPLAY, PARAMETER_CATEGORIES, formatParameterValue } from '../../services/constants';

function PredictionResult({ result, estimatedParams = null }) {
  if (!result) return null;

  const { prediction_label, confidence, probabilities, metadata } = result;
  const qualityConfig = QUALITY_COLORS[prediction_label];

  // Icon mapping
  const QualityIcon = {
    Good: CheckCircle2,
    Moderate: AlertTriangle,
    Poor: XCircle,
  }[prediction_label];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Result Card */}
      <div className={`card border-2 ${qualityConfig.border}`}>
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className={`p-4 rounded-full ${qualityConfig.bg}`}>
              <QualityIcon className={`w-8 h-8 ${qualityConfig.text}`} />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-900">
                Network Quality: <span className={qualityConfig.text}>{prediction_label}</span>
              </h3>
              <p className="text-gray-600 mt-1">
                Confidence: <span className="font-semibold">{(confidence * 100).toFixed(1)}%</span>
              </p>
            </div>
          </div>
          <span className="text-4xl">{qualityConfig.icon}</span>
        </div>

        {/* Probability Bars */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm font-medium text-gray-700">
            <span>Prediction Confidence</span>
            <span>{(confidence * 100).toFixed(1)}%</span>
          </div>
          {Object.entries(probabilities).map(([label, prob]) => (
            <div key={label} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-gray-700">{label}</span>
                <span className="text-gray-600">{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${
                    label === 'Good' ? 'bg-success-500' :
                    label === 'Moderate' ? 'bg-warning-500' : 'bg-danger-500'
                  }`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Metadata if available */}
        {metadata && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <p className="text-sm text-gray-600">
              {metadata.scenario_name && (
                <span className="font-medium">Scenario: {metadata.scenario_name}</span>
              )}
              {metadata.mode && (
                <span className="ml-4">Mode: <span className="font-medium">{metadata.mode}</span></span>
              )}
            </p>
          </div>
        )}
      </div>

      {/* Insight (Phân tích của AI) */}
      {result.insight && (
        <div className="card bg-blue-50 border border-blue-200 animate-fade-in">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="text-lg font-bold text-gray-900 mb-2">
                Phân tích của AI
              </h4>
              {/* Dùng dangerouslySetInnerHTML để render <strong> (in đậm) */}
              <p 
                className="text-sm text-blue-800 space-y-1"
                dangerouslySetInnerHTML={{ __html: result.insight }} 
              />
            </div>
          </div>
        </div>
      )}

      {/* Estimated Parameters (if Mode 1) */}
      {estimatedParams && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Activity className="w-5 h-5 text-primary-600" />
            <h4 className="text-lg font-bold text-gray-900">Estimated Network Parameters</h4>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Our AI estimated {Object.keys(estimatedParams).length} additional parameters based on your input.
          </p>

          {/* Group parameters by category */}
          <div className="space-y-6">
            {Object.entries(PARAMETER_CATEGORIES).map(([categoryKey, category]) => {
              const categoryParams = Object.entries(estimatedParams).filter(
                ([paramName]) => PARAMETER_DISPLAY[paramName]?.category === categoryKey
              );

              if (categoryParams.length === 0) return null;

              return (
                <div key={categoryKey}>
                  <div className="flex items-center space-x-2 mb-3">
                    <span className="text-xl">{category.icon}</span>
                    <h5 className="font-semibold text-gray-800">{category.label}</h5>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {categoryParams.map(([paramName, value]) => {
                      const config = PARAMETER_DISPLAY[paramName];
                      return (
                        <div key={paramName} className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                          <div className="text-xs text-gray-600 mb-1">{config.label}</div>
                          <div className="text-lg font-semibold text-gray-900">
                            {formatParameterValue(paramName, value)}
                            {config.unit && <span className="text-sm font-normal text-gray-600 ml-1">{config.unit}</span>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictionResult;