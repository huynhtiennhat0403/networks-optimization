import { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { Wifi, WifiOff, Zap, Server, Activity, Send, RefreshCw, Smartphone } from 'lucide-react';
import {
  USER_ACTIVITIES, DEVICE_TYPES, LOCATIONS, CONNECTION_TYPES, DEFAULT_VALUES, INPUT_RANGES,
} from '../services/constants';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PredictionResult from '../components/prediction/PredictionResult';

// --- Cấu hình Socket ---
const SERVER_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const SOCKETIO_PATH = "/ws/socket.io";

function RealtimeDashboard() {
  const [isWaitingForWorker, setIsWaitingForWorker] = useState(false); // Trạng thái chờ Worker
  
  const [contextData, setContextData] = useState({
    user_speed: DEFAULT_VALUES.user_speed,
    user_activity: DEFAULT_VALUES.user_activity,
    device_type: DEFAULT_VALUES.device_type,
    location: DEFAULT_VALUES.location,
    connection_type: DEFAULT_VALUES.connection_type,
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = io(SERVER_URL, {
      path: SOCKETIO_PATH,
      reconnectionAttempts: 5,
    });
    const socket = socketRef.current;

    socket.on('connect', () => {
      console.log('Socket: Connected');
      setIsConnected(true);
      setError(null);
    });

    socket.on('disconnect', () => {
      console.warn('Socket: Disconnected');
      setIsConnected(false);
      setIsWaitingForWorker(false);
      setError('Mất kết nối tới Server.');
    });

    socket.on('prediction_update', (data) => {
      console.log('Got result:', data);
      setResult(data);
      setError(null);
      setIsWaitingForWorker(false); // Đã có kết quả -> Dừng loading
    });
    
    socket.on('prediction_error', (data) => {
      setError(data.error);
      setIsWaitingForWorker(false);
    });

    return () => socket.disconnect();
  }, []);

  const handleStartPrediction = () => {
    if (socketRef.current && isConnected) {
      // 1. Gửi context lên server
      socketRef.current.emit('start_prediction', contextData);
      
      // 2. Chuyển sang trạng thái chờ Worker gửi metrics lên
      setResult(null);
      setError(null);
      setIsWaitingForWorker(true); 
    } else {
      setError("Chưa kết nối Server.");
    }
  };

  const handleChange = (field, value) => {
    let processedValue = value;
    if (field === 'user_speed') processedValue = parseFloat(value);
    setContextData(prev => ({ ...prev, [field]: processedValue }));
  };

  const StatusIcon = isConnected ? Wifi : WifiOff;
  const statusColor = isConnected ? 'text-success-600' : 'text-danger-600';

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8 animate-fade-in">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-3 bg-red-100 rounded-lg">
            <Activity className="w-8 h-8 text-red-600" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Real-time Monitor</h1>
            <p className="text-gray-600">Chế độ đo đạc tự động qua TCP Worker</p>
          </div>
        </div>
        
        <div className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
          <div className="flex items-center space-x-3">
            <Server className="w-5 h-5 text-gray-500" />
            <span className="font-medium text-gray-700">System Status:</span>
            <div className={`flex items-center space-x-1.5 ${statusColor}`}>
              <StatusIcon className="w-5 h-5" />
              <span className="font-semibold">{isConnected ? 'Server Online' : 'Server Offline'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Context */}
        <div className="card animate-slide-up h-fit">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
            <Smartphone className="w-5 h-5 mr-2" /> Bối cảnh thiết bị
          </h2>
          <p className="text-sm text-gray-600 mb-6">
            Cập nhật bối cảnh hiện tại. Hệ thống sẽ kết hợp với dữ liệu đo từ Worker để dự đoán.
          </p>
          
          <div className="space-y-5">
            {/* User Speed */}
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

            {/* Nút bấm */}
            <div className="pt-4 border-t border-gray-100">
              <button
                onClick={handleStartPrediction}
                disabled={!isConnected || isWaitingForWorker}
                className={`btn-primary w-full flex items-center justify-center space-x-2
                  ${(isWaitingForWorker || !isConnected) ? 'opacity-70 cursor-not-allowed' : ''}
                `}
              >
                {isWaitingForWorker ? (
                  <LoadingSpinner size="sm" text="" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                <span>{isWaitingForWorker ? 'Đang chờ Worker gửi dữ liệu...' : 'Cập nhật & Chờ kết quả'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Result Area */}
        <div className="lg:sticky lg:top-8 lg:self-start">
          {error && (
            <div className="p-4 mb-4 bg-red-50 text-red-700 rounded-lg border border-red-200">
              {error}
            </div>
          )}

          {isWaitingForWorker && (
            <div className="card bg-gray-50 text-center py-12 animate-pulse">
              <RefreshCw className="w-12 h-12 text-primary-400 mx-auto mb-4 animate-spin" />
              <h3 className="text-lg font-semibold text-gray-800">Đang chờ dữ liệu đo đạc...</h3>
              <p className="text-gray-500 mt-2 max-w-xs mx-auto">
                Server đã nhận bối cảnh. Đang đợi Worker (TCP) gửi gói tin Latency & Throughput tiếp theo.
              </p>
            </div>
          )}

          {!isWaitingForWorker && result && (
            <PredictionResult 
              result={result} 
              estimatedParams={result.metadata?.estimated_features_dict}
            />
          )}

          {!isWaitingForWorker && !result && (
            <div className="card bg-gray-50 text-center py-12">
              <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">
                Nhấn nút "Cập nhật" bên trái để bắt đầu phiên dự đoán mới.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RealtimeDashboard;