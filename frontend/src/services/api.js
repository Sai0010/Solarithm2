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
    { id: 1, name: 'Bulb', state: true, auto_controlled: true },
    { id: 2, name: 'Fan', state: false, auto_controlled: false },
    { id: 3, name: 'Grid Connection Relay', state: true, auto_controlled: true },
  ];
};

export const setRelayState = async (relayId, state, deviceId = 'solar_panel_01') => {
  try {
    // Log to backend; UI state is updated optimistically by callers
    await api.post('/control', { device_id: deviceId, relay_id: String(relayId), action: !!state });
  } catch (e) {
    // Swallow errors to prevent UI revert; log for diagnostics
    // eslint-disable-next-line no-console
    console.warn('setRelayState failed (non-fatal):', e?.message || e);
  }
};

export default api;