import { useState } from 'react';
import { Zap, Send, RefreshCw, Info, Battery, Gauge, Wifi } from 'lucide-react'; // icons mới
import { predictSimple } from '../services/api';
import { 
  USER_ACTIVITIES, 
  DEVICE_TYPES, 
  LOCATIONS, 
  CONNECTION_TYPES, 
  DEFAULT_VALUES,
  INPUT_RANGES,
  SIGNAL_BARS_MAP 
} from '../services/constants';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PredictionResult from '../components/prediction/PredictionResult';

function SmartInput() {
  const [formData, setFormData] = useState(DEFAULT_VALUES);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (field, value) => {
    // Chuyển đổi giá trị số nếu cần
    const numericFields = ['throughput', 'latency', 'user_speed', 'battery_level', 'signal_strength'];
    let processedValue = value;
    if (numericFields.includes(field)) {
      processedValue = parseFloat(value);
    }
    
    setFormData(prev => ({ ...prev, [field]: processedValue }));
    setError(null);
  };

  const handleReset = () => {
    setFormData(DEFAULT_VALUES);
    setResult(null);
    setError(null);
  };

  // 2.  hàm validate
  const validateInput = () => {
    const { throughput, latency, user_speed, battery_level } = formData;
    
    // Kiểm tra 4 trường người dùng nhập thủ công
    if (throughput < INPUT_RANGES.throughput.min || throughput > INPUT_RANGES.throughput.max) {
      return `Throughput phải từ ${INPUT_RANGES.throughput.min}-${INPUT_RANGES.throughput.max} Mbps`;
    }
    if (latency < INPUT_RANGES.latency.min || latency > INPUT_RANGES.latency.max) {
      return `Latency phải từ ${INPUT_RANGES.latency.min}-${INPUT_RANGES.latency.max} ms`;
    }
    if (user_speed < INPUT_RANGES.user_speed.min || user_speed > INPUT_RANGES.user_speed.max) {
      return `User Speed phải từ ${INPUT_RANGES.user_speed.min}-${INPUT_RANGES.user_speed.max} km/h`;
    }
    if (battery_level < INPUT_RANGES.battery_level.min || battery_level > INPUT_RANGES.battery_level.max) {
      return `Battery Level phải từ ${INPUT_RANGES.battery_level.min}-${INPUT_RANGES.battery_level.max} %`;
    }
    
    // signal_strength không cần validate vì nó là <select>
    
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
            <p className="text-gray-600">Nhập 5 thông số đo đạc, AI sẽ suy luận các thông số còn lại</p>
          </div>
        </div>
        
        <div className="flex items-start space-x-2 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-blue-800">
            Sử dụng ứng dụng (ví dụ: Speedtest) để lấy <b>Throughput (Tốc tải)</b> và <b>Latency (PING)</b>.
            Các thông số còn lại có thể xem trên thiết bị của bạn.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="card animate-slide-up">
          <h2 className="text-xl font-bold text-gray-900 mb-6">Thông số đo đạc</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 3. */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Thông số bắt buộc (5)
              </h3>
              
              {/* === User Speed === */}
              <div>
                <label className="label flex items-center space-x-1">
                  <Gauge className="w-4 h-4" />
                  <span>Tốc độ di chuyển (User Speed) <span className="text-danger-500">*</span></span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.user_speed}
                    onChange={(e) => handleChange('user_speed', e.target.value)}
                    className="input-field"
                    step={INPUT_RANGES.user_speed.step}
                    min={INPUT_RANGES.user_speed.min}
                    max={INPUT_RANGES.user_speed.max}
                    required
                  />
                  <span className="text-gray-600 text-sm whitespace-nowrap">
                    {INPUT_RANGES.user_speed.unit}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  (Nhập 0 nếu đứng yên)
                </p>
              </div>

              {/* === Battery Level === */}
              <div>
                <label className="label flex items-center space-x-1">
                  <Battery className="w-4 h-4" />
                  <span>Mức pin (Battery Level) <span className="text-danger-500">*</span></span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.battery_level}
                    onChange={(e) => handleChange('battery_level', e.target.value)}
                    className="input-field"
                    step={INPUT_RANGES.battery_level.step}
                    min={INPUT_RANGES.battery_level.min}
                    max={INPUT_RANGES.battery_level.max}
                    required
                  />
                  <span className="text-gray-600 text-sm whitespace-nowrap">
                    {INPUT_RANGES.battery_level.unit}
                  </span>
                </div>
              </div>

              {/* ===  Signal Strength === */}
              <div>
                <label className="label flex items-center space-x-1">
                  <Wifi className="w-4 h-4" />
                  <span>Vạch sóng (Signal Strength) <span className="text-danger-500">*</span></span>
                </label>
                <select
                  value={formData.signal_strength}
                  onChange={(e) => handleChange('signal_strength', e.target.value)}
                  className="input-field"
                >
                  {SIGNAL_BARS_MAP.map(bar => (
                    <option key={bar.value} value={bar.value}>
                      {bar.label} (Ước lượng {bar.value} dBm)
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  (Chọn vạch sóng di động 4G/5G của bạn)
                </p>
              </div>

              {/* === Throughput === */}
              <div>
                <label className="label">
                  Thông lượng (Throughput) <span className="text-danger-500">*</span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.throughput}
                    onChange={(e) => handleChange('throughput', e.target.value)}
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
                  (Dùng Speedtest, nhập <b>Tốc tải/Download</b>)
                </p>
              </div>

              {/* === Latency === */}
              <div>
                <label className="label">
                  Độ trễ (Latency) <span className="text-danger-500">*</span>
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={formData.latency}
                    onChange={(e) => handleChange('latency', e.target.value)}
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
                  (Dùng Speedtest, nhập <b>PING</b>)
                </p>
              </div>
            </div>

            {/* 4.  TIÊU ĐỀ Optional Context */}
            <div className="pt-6 border-t border-gray-200 space-y-4">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Bối cảnh (4)
              </h3>

              {/* User Activity */}
              <div>
                <label className="label">Hoạt động (User Activity)</label>
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
                <label className="label">Thiết bị (Device Type)</label>
                <select
                  value={formData.device_type}
                  onChange={(e) => handleChange('device_type', e.target.value)}
                  className="input-field"
                >
                  {/* Tự động dùng list DEVICE_TYPES mới (đã bỏ 'iot') */}
                  {DEVICE_TYPES.map(device => (
                    <option key={device.value} value={device.value}>
                      {device.icon} {device.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Location */}
              <div>
                <label className="label">Vị trí (Location)</label>
                <select
                  value={formData.location}
                  onChange={(e) => handleChange('location', e.target.value)}
                  className="input-field"
                >
                  {/* LOCATIONS */}
                  {LOCATIONS.map(loc => (
                    <option key={loc.value} value={loc.value}>
                      {loc.icon} {loc.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Connection Type */}
              <div>
                <label className="label">Loại kết nối (Connection Type)</label>
                <select
                  value={formData.connection_type}
                  onChange={(e) => handleChange('connection_type', e.target.value)}
                  className="input-field"
                >
                  {/* CONNECTION_TYPES */}
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
                    <span>Đang dự đoán...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Dự đoán chất lượng</span>
                  </>
                )}
              </button>
              
              <button
                type="button"
                onClick={handleReset}
                className="btn-secondary flex items-center space-x-2"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Làm mới</span>
              </button>
            </div>
          </form>
        </div>

        {/* Result Section */}
        <div id="result-section" className="lg:sticky lg:top-8 lg:self-start">
          {loading && (
            <div className="card animate-fade-in">
              <LoadingSpinner text="Đang phân tích thông số mạng..." />
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
                  Điền 5 thông số và bấm "Dự đoán" để xem kết quả
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