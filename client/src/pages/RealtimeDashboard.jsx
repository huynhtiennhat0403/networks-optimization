import { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { Wifi, WifiOff, Zap, Server, Activity } from 'lucide-react';
import {
  USER_ACTIVITIES,
  DEVICE_TYPES,
  LOCATIONS,
  CONNECTION_TYPES,
  DEFAULT_VALUES,
  INPUT_RANGES,
} from '../services/constants';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PredictionResult from '../components/prediction/PredictionResult';

// --- Cấu hình Socket ---
const SERVER_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const SOCKETIO_PATH = "/ws/socket.io";

function RealtimeDashboard() {
  const [contextData, setContextData] = useState({
    // 5 thông số bối cảnh
    user_speed: DEFAULT_VALUES.user_speed,
    user_activity: DEFAULT_VALUES.user_activity,
    device_type: DEFAULT_VALUES.device_type,
    location: DEFAULT_VALUES.location,
    connection_type: DEFAULT_VALUES.connection_type,
  });
  
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isWorkerConnected, setIsWorkerConnected] = useState(false);
  
  // Dùng useRef để socket tồn tại qua các lần render
  const socketRef = useRef(null);

  useEffect(() => {
    // --- 1. Khởi tạo kết nối Socket ---
    socketRef.current = io(SERVER_URL, {
      path: SOCKETIO_PATH,
      reconnectionAttempts: 5,
    });

    const socket = socketRef.current;

    // --- 2. Lắng nghe các sự kiện từ Server ---
    socket.on('connect', () => {
      console.log('Socket.IO: Đã kết nối tới server');
      setIsConnected(true);
      // Gửi bối cảnh ban đầu ngay khi kết nối
      socket.emit('context_update', contextData);
    });

    socket.on('disconnect', () => {
      console.warn('Socket.IO: Đã ngắt kết nối');
      setIsConnected(false);
      setIsWorkerConnected(false); // Khi mất kết nối, reset trạng thái worker
      setError('Đã mất kết nối tới server. Đang thử kết nối lại...');
    });

    // Sự kiện: Server phát (broadcast) kết quả dự đoán
    socket.on('prediction_update', (predictionData) => {
      console.log('Nhận [prediction_update]:', predictionData);
      setResult(predictionData);
      setError(null);
      // Đánh dấu là worker đã gửi dữ liệu
      if (!isWorkerConnected) setIsWorkerConnected(true);
    });
    
    // Sự kiện: Server báo lỗi
    socket.on('prediction_error', (errorData) => {
      console.error('Nhận [prediction_error]:', errorData);
      setError(errorData.error || 'Lỗi từ server');
    });

    // Cleanup: Ngắt kết nối khi component bị unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []); // Chỉ chạy 1 lần khi mount

  // --- 3. Gửi 'context_update' khi người dùng thay đổi lựa chọn ---
  useEffect(() => {
    if (socketRef.current && isConnected) {
      console.log('Gửi [context_update]:', contextData);
      socketRef.current.emit('context_update', contextData);
    }
  }, [contextData, isConnected]); // Chạy lại khi contextData hoặc isConnected thay đổi

  // Hàm xử lý khi form thay đổi
  const handleChange = (field, value) => {
    let processedValue = value;
    if (field === 'user_speed') {
      processedValue = parseFloat(value);
    }
    setContextData(prev => ({ ...prev, [field]: processedValue }));
    setError(null);
  };

  // Trạng thái kết nối
  const StatusIcon = isConnected ? Wifi : WifiOff;
  const statusColor = isConnected ? 'text-success-600' : 'text-danger-600';
  const statusText = isConnected ? 'Đã kết nối' : 'Đã ngắt kết nối';

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-3 bg-primary-100 rounded-lg">
            <Activity className="w-8 h-8 text-primary-600" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Real-time Dashboard</h1>
            <p className="text-gray-600">Theo dõi chất lượng mạng tự động (Mode Tự động)</p>
          </div>
        </div>
        
        {/* Status Bar */}
        <div className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
          <div className="flex items-center space-x-3">
            <Server className="w-5 h-5 text-gray-500" />
            <span className="font-medium text-gray-700">Server Status:</span>
            <div className={`flex items-center space-x-1.5 ${statusColor}`}>
              <StatusIcon className="w-5 h-5" />
              <span className="font-semibold">{statusText}</span>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Zap className="w-5 h-5 text-gray-500" />
            <span className="font-medium text-gray-700">Worker Status:</span>
            {isWorkerConnected ? (
              <span className="font-semibold text-success-600">Đang gửi dữ liệu</span>
            ) : (
              <span className="font-semibold text-warning-600">Đang chờ dữ liệu...</span>
            )}
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Form (5 thông số bối cảnh) */}
        <div className="card animate-slide-up">
          <h2 className="text-xl font-bold text-gray-900 mb-6">
            Thông số bối cảnh (Context)
          </h2>
          <p className="text-sm text-gray-600 mb-6">
            Chọn các bối cảnh hiện tại của bạn. Kết quả sẽ tự động cập nhật 
            khi Worker (script Python) gửi dữ liệu đo đạc mới.
          </p>
          
          <div className="space-y-6">
            {/* === User Speed === */}
            <div>
              <label className="label">Tốc độ di chuyển (User Speed) *</label>
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  value={contextData.user_speed}
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
            </div>
            
            {/* User Activity */}
            <div>
              <label className="label">Hoạt động (User Activity)</label>
              <select
                value={contextData.user_activity}
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
                value={contextData.device_type}
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
              <label className="label">Vị trí (Location)</label>
              <select
                value={contextData.location}
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
              <label className="label">Loại kết nối (Connection Type)</label>
              <select
                value={contextData.connection_type}
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
        </div>

        {/* Result Section */}
        <div id="result-section" className="lg:sticky lg:top-8 lg:self-start">
          {error && (
            <div className="card bg-danger-50 border-danger-200 mb-4">
              <p className="text-sm text-danger-700">{error}</p>
            </div>
          )}
          
          {!result && (
            <div className="card bg-gray-50 text-center">
              <div className="py-8">
                {!isConnected ? (
                  <LoadingSpinner text="Đang kết nối tới server..." />
                ) : !isWorkerConnected ? (
                  <LoadingSpinner text="Đang chờ dữ liệu từ Worker (worker.py)..." />
                ) : (
                  <LoadingSpinner text="Đang xử lý..." />
                )}
                <p className="text-xs text-gray-500 mt-4">
                  Hãy đảm bảo bạn đã chạy cả Server (main.py) và Worker (worker.py).
                </p>
              </div>
            </div>
          )}
          
          {result && (
            <PredictionResult 
              result={result} 
              // Truyền vào các thông số đã ước tính
              estimatedParams={result.metadata?.estimated_features_dict}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default RealtimeDashboard;