import axios from 'axios';

// Use CRA proxy by default ("proxy" in package.json). Allow override via env.
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// Health
export const getHealth = async () => {
  const { data } = await api.get('/health');
  return data;
};

// Latest reading for a device
export const getLatestReading = async (deviceId) => {
  const { data } = await api.get('/latest', { params: { device_id: deviceId } });
  return data;
};

// Power forecast
export const getPowerForecast = async (deviceId, horizon = 6) => {
  const { data } = await api.get('/forecast', { params: { device_id: deviceId, horizon } });
  return data;
};

// Relay helpers (no real backend yet). Use backend /control to log, simulate reads.
export const getRelayStatus = async () => {
  // Simulated relay state
  return [
    { id: 1, name: 'Main Power Relay', state: true, auto_controlled: true },
    { id: 2, name: 'Battery Charging Relay', state: false, auto_controlled: true },
    { id: 3, name: 'Grid Connection Relay', state: true, auto_controlled: true },
    { id: 4, name: 'Auxiliary System Relay', state: false, auto_controlled: false },
  ];
};

export const setRelayState = async (relayId, state, deviceId = 'solar_panel_01') => {
  try {
    // Log to backend; UI state is updated optimistically by callers
    await api.post('/control', { device_id: deviceId, relay_id: String(relayId), action: !!state });
  } catch (e) {
    // Non-fatal for UI; surface to caller if needed
    throw e;
  }
};

export default api;