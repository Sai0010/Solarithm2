import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = {
  // Sensor readings
  getSensorReadings: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/sensors/latest`);
      return response.data;
    } catch (error) {
      console.error('Error fetching sensor readings:', error);
      throw error;
    }
  },
  
  // Relay controls
  getRelayStatus: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/relays`);
      return response.data;
    } catch (error) {
      console.error('Error fetching relay status:', error);
      throw error;
    }
  },
  
  setRelayState: async (relayId, state) => {
    try {
      const response = await axios.post(`${API_URL}/api/relays`, {
        relay_id: relayId,
        state: state
      });
      return response.data;
    } catch (error) {
      console.error('Error setting relay state:', error);
      throw error;
    }
  },
  
  // Power forecasting
  getPowerForecast: async (hours = 24) => {
    try {
      const response = await axios.get(`${API_URL}/api/forecast?hours=${hours}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching power forecast:', error);
      throw error;
    }
  },
  
  // Health check
  getSystemHealth: async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      return response.data;
    } catch (error) {
      console.error('Error checking system health:', error);
      throw error;
    }
  }
};

export default api;