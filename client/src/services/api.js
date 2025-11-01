import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor (for debugging)
api.interceptors.request.use(
  (config) => {
    // Log request
    if (config.data) {
      console.log(`[API] ${config.method.toUpperCase()} ${config.url}`, config.data);
    } else {
      console.log(`[API] ${config.method.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor (for error handling)
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('[API] Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ==================== API METHODS ====================

/**
 * Mode 1: Smart Input Prediction
 * User cung cấp 5 thông số đo đạc + 4 bối cảnh
 */
export const predictSimple = async (data) => {
  // 'data' object sẽ chứa tất cả 9 trường từ form
  const payload = {
    // 5 thông số đo đạc
    user_speed: data.user_speed,
    battery_level: data.battery_level,
    throughput: data.throughput,
    latency: data.latency,
    signal_strength: data.signal_strength,
    
    // 4 thông số bối cảnh
    user_activity: data.user_activity,
    device_type: data.device_type,
    location: data.location,
    connection_type: data.connection_type,
  };
  
  const response = await api.post('/predict/simple', payload);
  return response.data;
};

/**
 * Mode 2: Scenario Selection 
 */
export const predictScenario = async (scenarioId) => {
  const response = await api.post('/predict/scenario', {
    scenario_id: scenarioId,
  });
  return response.data;
};

/**
 * Get all available scenarios 
 */
export const getScenarios = async () => {
  // Sửa: Dùng endpoint /scenarios/list
  const response = await api.get('/scenarios/list');
  return response.data;
};

/**
 * Get specific scenario details 
 */
export const getScenarioDetails = async (scenarioId) => {
  const response = await api.get(`/scenarios/${scenarioId}`);
  return response.data;
};

/**
 * Health check 
 */
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;