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
    console.log(`[API] ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor (for error handling)
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response:`, response.data);
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
 * User provides 5 basic metrics, server estimates 10 more
 */
export const predictSimple = async (data) => {
  const response = await api.post('/predict/simple', {
    throughput: data.throughput,
    latency: data.latency,
    signal_strength: data.signal_strength,
    user_activity: data.user_activity || 'browsing',
    device_type: data.device_type || 'laptop',
    location: data.location || 'home',
    connection_type: data.connection_type || 'wifi',
  });
  return response.data;
};

/**
 * Mode 2: Scenario Selection
 * User selects a predefined scenario
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