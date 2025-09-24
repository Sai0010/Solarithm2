import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Sensor data endpoints
export const getSensorReadings = async (params) => {
  try {
    const response = await api.get('/sensor-readings', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching sensor readings:', error);
    throw error;
  }
};

// Power forecast endpoints
export const getPowerForecast = async (deviceId, horizon = 6) => {
  try {
    const response = await api.get('/forecast', { 
      params: { 
        device_id: deviceId,
        horizon: horizon 
      } 
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching power forecast:', error);
    throw error;
  }
};

// System status endpoints
export const getSystemStatus = async () => {
  try {
    const response = await api.get('/system/status');
    return response.data;
  } catch (error) {
    console.error('Error fetching system status:', error);
    throw error;
  }
};

// Relay control endpoints
export const getRelayStatus = async () => {
  try {
    const response = await api.get('/relay/status');
    return response.data;
  } catch (error) {
    console.error('Error fetching relay status:', error);
    throw error;
  }
};

export const setRelayState = async (relayId, state) => {
  try {
    const response = await api.post('/relay/control', { relay_id: relayId, state });
    return response.data;
  } catch (error) {
    console.error('Error setting relay state:', error);
    throw error;
  }
};

export default api;