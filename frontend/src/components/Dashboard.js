import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  const [sensorData, setSensorData] = useState(null);
  const [relayData, setRelayData] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch all required data in parallel
        const [sensorResponse, relayResponse, healthResponse] = await Promise.all([
          api.getSensorReadings(),
          api.getRelayStatus(),
          api.getSystemHealth()
        ]);
        
        setSensorData(sensorResponse);
        setRelayData(relayResponse);
        setSystemHealth(healthResponse);
        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="loading">Loading dashboard data...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="dashboard">
      <h2>Solar Monitoring Dashboard</h2>
      
      {/* System Status Card */}
      <div className="dashboard-grid">
        <div className="dashboard-card">
          <h3>System Status</h3>
          {systemHealth && (
            <div>
              <p>Status: <span className={systemHealth.status === 'healthy' ? 'status-ok' : 'status-error'}>
                {systemHealth.status === 'healthy' ? 'Online' : 'Error'}
              </span></p>
              <p>API Version: {systemHealth.version || 'Unknown'}</p>
              <p>Database: {systemHealth.database_status === 'connected' ? 'Connected' : 'Disconnected'}</p>
            </div>
          )}
        </div>

        {/* Current Power Card */}
        <div className="dashboard-card">
          <h3>Current Power</h3>
          {sensorData && (
            <div>
              <div className="sensor-reading">
                <div className="label">Voltage:</div>
                <div className="value">{sensorData.voltage_v.toFixed(2)} V</div>
              </div>
              <div className="sensor-reading">
                <div className="label">Current:</div>
                <div className="value">{sensorData.current_a.toFixed(2)} A</div>
              </div>
              <div className="sensor-reading">
                <div className="label">Power:</div>
                <div className="value">{(sensorData.voltage_v * sensorData.current_a).toFixed(2)} W</div>
              </div>
              <div className="sensor-reading">
                <div className="label">Last Updated:</div>
                <div className="value">{new Date(sensorData.ts).toLocaleTimeString()}</div>
              </div>
            </div>
          )}
          <Link to="/sensors" className="btn">View Details</Link>
        </div>

        {/* Environmental Data Card */}
        <div className="dashboard-card">
          <h3>Environmental Data</h3>
          {sensorData && (
            <div>
              <div className="sensor-reading">
                <div className="label">Temperature:</div>
                <div className="value">{sensorData.temp_c.toFixed(1)} °C</div>
              </div>
              <div className="sensor-reading">
                <div className="label">Humidity:</div>
                <div className="value">{sensorData.humidity_pct.toFixed(1)} %</div>
              </div>
              <div className="sensor-reading">
                <div className="label">Light Level:</div>
                <div className="value">{sensorData.lux.toFixed(0)} lux</div>
              </div>
            </div>
          )}
        </div>

        {/* Relay Status Card */}
        <div className="dashboard-card">
          <h3>Relay Status</h3>
          {relayData && relayData.length > 0 ? (
            <div className="relay-status-grid">
              {relayData.map((relay) => (
                <div key={relay.relay_id} className="relay-status-item">
                  <span>Relay {relay.relay_id + 1}: </span>
                  <span className={relay.state ? 'status-on' : 'status-off'}>
                    {relay.state ? 'ON' : 'OFF'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p>No relay data available</p>
          )}
          <Link to="/relays" className="btn">Control Relays</Link>
        </div>
      </div>
      
      {/* Power Forecast Preview */}
      <div className="forecast-preview">
        <h3>Power Forecast</h3>
        <p>View detailed power forecasts based on our AI model to optimize your solar energy usage.</p>
        <Link to="/forecast" className="btn">View Forecast</Link>
      </div>
    </div>
  );
};

export default Dashboard;