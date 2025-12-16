import { CheckCircle2, AlertTriangle, XCircle, Activity, Info, Lightbulb } from 'lucide-react'; // Thêm Lightbulb
import { QUALITY_COLORS, PARAMETER_DISPLAY, PARAMETER_CATEGORIES, formatParameterValue } from '../../services/constants';

function PredictionResult({ result, estimatedParams = null }) {
  if (!result) return null;

  const { prediction_label, confidence, probabilities, metadata, insight } = result;
  const qualityConfig = QUALITY_COLORS[prediction_label] || QUALITY_COLORS.Moderate;

  // Icon mapping
  const QualityIcon = {
    Good: CheckCircle2,
    Moderate: AlertTriangle,
    Poor: XCircle,
  }[prediction_label] || AlertTriangle;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 1. Main Result Card */}
      <div className={`card border-2 ${qualityConfig.border} shadow-lg`}>
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
          <span className="text-4xl filter drop-shadow-sm">{qualityConfig.icon}</span>
        </div>

        {/* Probability Bars */}
        <div className="space-y-3 bg-gray-50 p-4 rounded-lg">
           {/* ... (Giữ nguyên code vẽ bar chart cũ của bạn) ... */}
           {Object.entries(probabilities).map(([label, prob]) => (
            <div key={label} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-gray-700">{label}</span>
                <span className="text-gray-600">{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    label === 'Good' ? 'bg-success-500' :
                    label === 'Moderate' ? 'bg-warning-500' : 'bg-danger-500'
                  }`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 2. AI Insight / Recommendation (Quan trọng) */}
      {insight && (
        <div className="card bg-blue-50 border border-blue-200 animate-slide-up shadow-sm">
          <div className="flex items-start space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
               <Lightbulb className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h4 className="text-lg font-bold text-gray-900 mb-2">
                AI Recommendation
              </h4>
              <div 
                className="text-gray-800 space-y-1 leading-relaxed"
                dangerouslySetInnerHTML={{ __html: insight }} 
              />
            </div>
          </div>
        </div>
      )}

      {/* 3. Estimated Parameters */}
      {estimatedParams && (
        <div className="card border border-gray-200">
          <div className="flex items-center space-x-2 mb-4 border-b pb-2">
            <Activity className="w-5 h-5 text-primary-600" />
            <h4 className="text-lg font-bold text-gray-900">Physical Parameters Analysis</h4>
          </div>
          
          <div className="space-y-6">
            {Object.entries(PARAMETER_CATEGORIES).map(([categoryKey, category]) => {
              // Filter các params thuộc category này VÀ có trong estimatedParams
              const categoryParams = Object.entries(estimatedParams).filter(
                ([paramName]) => PARAMETER_DISPLAY[paramName]?.category === categoryKey
              );

              if (categoryParams.length === 0) return null;

              return (
                <div key={categoryKey}>
                  <div className="flex items-center space-x-2 mb-3">
                    <span className="text-xl">{category.icon}</span>
                    <h5 className="font-semibold text-gray-700 text-sm uppercase tracking-wide">{category.label}</h5>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {categoryParams.map(([paramName, value]) => {
                      const config = PARAMETER_DISPLAY[paramName];
                      return (
                        <div key={paramName} className="bg-white rounded-lg p-3 border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
                          <div className="text-xs text-gray-500 mb-1 truncate" title={config.label}>{config.label}</div>
                          <div className="text-lg font-bold text-gray-900">
                            {formatParameterValue(paramName, value)}
                            {config.unit && <span className="text-sm font-normal text-gray-500 ml-1">{config.unit}</span>}
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