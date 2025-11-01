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

// Input validation ranges (based on actual data)
export const INPUT_RANGES = {
  throughput: { min: 1, max: 100, unit: 'Mbps', step: 0.1 },
  latency: { min: 1, max: 100, unit: 'ms', step: 0.1 },
  signal_strength: { min: -100, max: -40, unit: 'dBm', step: 1 },
};

// Default values for Mode 1
export const DEFAULT_VALUES = {
  throughput: 50.0,
  latency: 85.0,
  signal_strength: -85.0, // Default là 3 vạch
  user_speed: 0,
  battery_level: 100,
  user_activity: 'browsing',
  device_type: 'phone', 
  location: 'home',
  connection_type: '4g', 
};

// Network parameter display configuration
export const PARAMETER_DISPLAY = {
  'User Speed (m/s)': { label: 'User Speed', unit: 'm/s', decimals: 2, category: 'mobility' },
  'User Direction (degrees)': { label: 'Direction', unit: '°', decimals: 1, category: 'mobility' },
  'Handover Events': { label: 'Handovers', unit: '', decimals: 0, category: 'mobility' },
  'Distance from Base Station (m)': { label: 'Distance to BS', unit: 'm', decimals: 1, category: 'signal' },
  'Signal Strength (dBm)': { label: 'Signal Strength', unit: 'dBm', decimals: 1, category: 'signal' },
  'SNR (dB)': { label: 'SNR', unit: 'dB', decimals: 2, category: 'signal' },
  'BER': { label: 'Bit Error Rate', unit: '', decimals: 4, category: 'quality' },
  'Modulation Scheme': { label: 'Modulation', unit: '', decimals: 0, category: 'quality' },
  'PDR (%)': { label: 'Packet Delivery', unit: '%', decimals: 1, category: 'quality' },
  'Network Congestion': { label: 'Congestion', unit: '', decimals: 0, category: 'quality' },
  'Throughput (Mbps)': { label: 'Throughput', unit: 'Mbps', decimals: 1, category: 'performance' },
  'Latency (ms)': { label: 'Latency', unit: 'ms', decimals: 1, category: 'performance' },
  'Retransmission Count': { label: 'Retransmissions', unit: '', decimals: 0, category: 'performance' },
  'Power Consumption (mW)': { label: 'Power Usage', unit: 'mW', decimals: 1, category: 'power' },
  'Battery Level (%)': { label: 'Battery', unit: '%', decimals: 1, category: 'power' },
  'Transmission Power (dBm)': { label: 'TX Power', unit: 'dBm', decimals: 1, category: 'power' },
};

// Parameter categories for organized display
export const PARAMETER_CATEGORIES = {
  mobility: { label: 'Mobility', icon: '🚶', color: 'blue' },
  signal: { label: 'Signal Quality', icon: '📡', color: 'green' },
  quality: { label: 'Connection Quality', icon: '🎯', color: 'purple' },
  performance: { label: 'Performance', icon: '⚡', color: 'yellow' },
  power: { label: 'Power & Battery', icon: '🔋', color: 'red' },
};

// Network Congestion mapping
export const NETWORK_CONGESTION_MAP = {
  0: 'Low',
  1: 'Medium',
  2: 'High',
  'Low': 0,
  'Medium': 1,
  'High': 2,
};

// Modulation schemes
export const MODULATION_SCHEMES = ['BPSK', 'QPSK', '16-QAM', '64-QAM'];

// Helper function to format parameter value
export const formatParameterValue = (paramName, value) => {
  const config = PARAMETER_DISPLAY[paramName];
  if (!config) return value;
  
  if (paramName === 'Network Congestion') {
    return typeof value === 'number' ? NETWORK_CONGESTION_MAP[value] : value;
  }
  
  if (paramName === 'Modulation Scheme') {
    return value;
  }
  
  if (config.decimals === 0) {
    return Math.round(value);
  }
  
  return Number(value).toFixed(config.decimals);
};