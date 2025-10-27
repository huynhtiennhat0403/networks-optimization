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
  },
  Moderate: {
    bg: 'bg-warning-50',
    text: 'text-warning-700',
    border: 'border-warning-200',
  },
  Good: {
    bg: 'bg-success-50',
    text: 'text-success-700',
    border: 'border-success-200',
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
  { value: 'iot', label: 'IoT Device', icon: '🔌' },
];

// Location types
export const LOCATIONS = [
  { value: 'home', label: 'Home', icon: '🏠' },
  { value: 'office', label: 'Office', icon: '🏢' },
  { value: 'public', label: 'Public Place', icon: '🏪' },
  { value: 'outdoor', label: 'Outdoor', icon: '🌳' },
  { value: 'vehicle', label: 'Vehicle', icon: '🚗' },
  { value: 'event', label: 'Crowded Event', icon: '🎉' },
];

// Connection types
export const CONNECTION_TYPES = [
  { value: 'wifi', label: 'WiFi', icon: '📶' },
  { value: '4g', label: '4G/LTE', icon: '📡' },
  { value: '5g', label: '5G', icon: '🚀' },
  { value: 'ethernet', label: 'Ethernet', icon: '🔌' },
];

// Input validation ranges
export const INPUT_RANGES = {
  throughput: { min: 1, max: 100, unit: 'Mbps' },
  latency: { min: 1, max: 100, unit: 'ms' },
  signal_strength: { min: -100, max: -40, unit: 'dBm' },
};

// Default values for Mode 1
export const DEFAULT_VALUES = {
  throughput: 45,
  latency: 25,
  signal_strength: -65,
  user_activity: 'browsing',
  device_type: 'laptop',
  location: 'home',
  connection_type: 'wifi',
};