// src/services/constants.js

// Quality labels mapping
export const QUALITY_LABELS = {
  0: 'Poor',
  1: 'Moderate',
  2: 'Good',
};

// Quality colors for badges
export const QUALITY_COLORS = {
  Poor: {
    bg: 'bg-danger-50',
    text: 'text-danger-700',
    border: 'border-danger-200',
    icon: '❌',
  },
  Moderate: {
    bg: 'bg-warning-50',
    text: 'text-warning-700',
    border: 'border-warning-200',
    icon: '⚠️',
  },
  Good: {
    bg: 'bg-success-50',
    text: 'text-success-700',
    border: 'border-success-200',
    icon: '✅',
  },
};

// User activity options
export const USER_ACTIVITIES = [
  { value: 'browsing', label: 'Web Browsing', icon: '🌐' },
  { value: 'streaming', label: 'Video Streaming', icon: '📹' },
  { value: 'gaming', label: 'Online Gaming', icon: '🎮' },
  { value: 'downloading', label: 'Downloading Files', icon: '📥' },
  { value: 'video_call', label: 'Video Call', icon: '📞' },
];

// Device types 
export const DEVICE_TYPES = [
  { value: 'phone', label: 'Smartphone', icon: '📱' },
  { value: 'laptop', label: 'Laptop', icon: '💻' },
  { value: 'tablet', label: 'Tablet', icon: '📲' },
];

// Location types 
export const LOCATIONS = [
  { value: 'home', label: 'Home', icon: '🏠' },
  { value: 'office', label: 'Office', icon: '🏢' },
  { value: 'outdoor', label: 'Outdoor', icon: '🌳' },
  { value: 'vehicle', label: 'Vehicle', icon: '🚗' },
  { value: 'event', label: 'Crowded Event', icon: '🎉' },
];

// Connection types 
export const CONNECTION_TYPES = [
  { value: '4g', label: '4G/LTE', icon: '📡' },
  { value: '5g', label: '5G', icon: '🚀' },
];

// Bảng quy đổi Vạch sóng (Update dBm mapping cho chuẩn hơn)
export const SIGNAL_BARS_MAP = [
  { label: '★☆☆☆ (1 vạch - Rất yếu)', value: -105.0 }, 
  { label: '★★☆☆ (2 vạch - Yếu)', value: -95.0 },     
  { label: '★★★☆ (3 vạch - Trung bình)', value: -85.0 }, 
  { label: '★★★★ (4 vạch - Mạnh)', value: -65.0 }       
];

// Input validation ranges
export const INPUT_RANGES = {
  throughput: { min: 0, max: 200, unit: 'Mbps', step: 0.1 }, 
  latency: { min: 1, max: 500, unit: 'ms', step: 0.1 },     
  user_speed: { min: 0, max: 120, unit: 'km/h', step: 1 },  
  battery_level: { min: 1, max: 100, unit: '%', step: 1 },  
};

// Default values
export const DEFAULT_VALUES = {
  throughput: 45.5,
  latency: 50.0,
  signal_strength: -85.0, 
  user_speed: 10,
  battery_level: 80,
  user_activity: 'streaming',
  device_type: 'laptop', 
  location: 'home',
  connection_type: '4g', 
};

// --- CẤU HÌNH HIỂN THỊ THAM SỐ ---
export const PARAMETER_DISPLAY = {
    // User inputs (5)
    'Throughput (Mbps)': { label: 'Throughput', unit: 'Mbps', category: 'performance', decimals: 1 },
    'Latency (ms)': { label: 'Latency', unit: 'ms', category: 'performance', decimals: 1 },
    'User Speed (m/s)': { label: 'Speed', unit: 'm/s', category: 'mobility', decimals: 2 },
    'Signal Strength (dBm)': { label: 'Signal', unit: 'dBm', category: 'signal', decimals: 1 },
    'Battery Level (%)': { label: 'Battery', unit: '%', category: 'power', decimals: 0 },
    
    // Estimated physical features (8)
    'Network Congestion': { label: 'Network Congestion', category: 'environment', decimals: 0 },
    'Distance from Base Station (m)': { label: 'Distance to Tower', unit: 'm', category: 'environment', decimals: 1 },
    'Handover Events': { label: 'Handover Events', unit: 'times', category: 'mobility', decimals: 0 },
    'Power Consumption (mW)': { label: 'Power Usage', unit: 'mW', category: 'power', decimals: 1 },
    'Transmission Power (dBm)': { label: 'TX Power', unit: 'dBm', category: 'power', decimals: 1 },
};

export const PARAMETER_CATEGORIES = {
  performance: { label: 'Input Metrics', icon: '⚡', color: 'blue' },
  environment: { label: 'Environment', icon: '🌍', color: 'green' },
  mobility: { label: 'Mobility', icon: '🚶', color: 'yellow' },
  power: { label: 'Power & Device', icon: '🔋', color: 'red' },
  signal: { label: 'Signal', icon: '📡', color: 'purple' },
};

export const formatParameterValue = (paramName, value) => {
  const config = PARAMETER_DISPLAY[paramName];
  if (!config) return value;
  
  if (paramName === 'Network Congestion') {
     // Backend trả về 'Low', 'Medium', 'High' hoặc số 1,2,3
     // Nếu là số thì map, nếu là chữ thì giữ nguyên
     const map = {1: 'Low', 2: 'Medium', 3: 'High'};
     return map[value] || value;
  }
  
  if (config.decimals === 0) {
    return Math.round(value);
  }
  
  return Number(value).toFixed(config.decimals);
};