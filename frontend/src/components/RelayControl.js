import React, { useState, useEffect } from 'react';
import { getRelayStatus, setRelayState } from '../services/api';

const RelayControl = () => {
  const [relays, setRelays] = useState([]);
  const [mode, setMode] = useState('auto');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch relay states on component mount and periodically
  useEffect(() => {
    const fetchRelayStates = async () => {
      try {
        setLoading(true);
        const data = await getRelayStatus();
        setRelays(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching relay states:', err);
        setError('Failed to load relay states. Please try again later.');
        // Use simulated data for development if API fails
        setRelays([
          { id: 1, name: 'Main Power Relay', state: true, auto_controlled: true },
          { id: 2, name: 'Battery Charging Relay', state: false, auto_controlled: true },
          { id: 3, name: 'Grid Connection Relay', state: true, auto_controlled: true },
          { id: 4, name: 'Auxiliary System Relay', state: false, auto_controlled: false }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchRelayStates();
    const interval = setInterval(fetchRelayStates, 10000); // Refresh every 10 seconds
    
    return () => clearInterval(interval);
  }, []);

  // Toggle relay state
  const toggleRelay = async (id) => {
    try {
      const relay = relays.find(r => r.id === id);
      if (!relay) return;
      
      // Optimistic update
      setRelays(relays.map(r => 
        r.id === id ? { ...r, state: !r.state } : r
      ));
      
      // API call to update relay state
      await setRelayState(id, !relay.state);
    } catch (err) {
      console.error(`Error toggling relay ${id}:`, err);
      setError(`Failed to toggle relay. Please try again.`);
      
      // Revert optimistic update on error
      fetchRelayStates();
    }
  };

  // Toggle control mode (auto/manual)
  const toggleMode = async (id) => {
    try {
      const relay = relays.find(r => r.id === id);
      if (!relay) return;
      
      // Optimistic update
      setRelays(relays.map(r => 
        r.id === id ? { ...r, auto_controlled: !r.auto_controlled } : r
      ));
      
      // API call to update control mode
      // No backend endpoint yet; keep UI-only for mode
    } catch (err) {
      console.error(`Error changing mode for relay ${id}:`, err);
      setError(`Failed to change relay mode. Please try again.`);
      
      // Revert optimistic update on error
      fetchRelayStates();
    }
  };

  // Set global mode for all relays
  const setGlobalMode = async (newMode) => {
    try {
      setMode(newMode);
      
      // API call to update all relays
      // No backend endpoint yet; keep UI-only for global mode
      
      // Update local state
      setRelays(relays.map(r => ({ 
        ...r, 
        auto_controlled: newMode === 'auto' 
      })));
    } catch (err) {
      console.error('Error setting global mode:', err);
      setError(`Failed to set ${newMode} mode. Please try again.`);
    }
  };

  // Helper function to fetch relay states
  const fetchRelayStates = async () => {
    try {
      const data = await getRelayStatus();
      setRelays(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching relay states:', err);
      // Don't set error or simulated data here to avoid overriding user actions
    }
  };

  if (loading && relays.length === 0) {
    return <div className="loading">Loading relay status...</div>;
  }

  return (
    <div className="relay-control-container">
      <h2>Relay Control Panel</h2>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="mode-selector">
        <span>Control Mode: </span>
        <button 
          className={`mode-button ${mode === 'auto' ? 'active' : ''}`}
          onClick={() => setGlobalMode('auto')}
        >
          Automatic
        </button>
        <button 
          className={`mode-button ${mode === 'manual' ? 'active' : ''}`}
          onClick={() => setGlobalMode('manual')}
        >
          Manual
        </button>
      </div>
      
      <div className="relays-grid">
        {relays.map(relay => (
          <div key={relay.id} className="relay-card">
            <h3>{relay.name}</h3>
            <div className="relay-status">
              Status: <span className={relay.state ? 'on' : 'off'}>
                {relay.state ? 'ON' : 'OFF'}
              </span>
            </div>
            <div className="relay-mode">
              Mode: {relay.auto_controlled ? 'Automatic' : 'Manual'}
            </div>
            <div className="relay-controls">
              <button 
                className={`toggle-button ${relay.state ? 'on' : 'off'}`}
                onClick={() => toggleRelay(relay.id)}
                disabled={relay.auto_controlled}
              >
                {relay.state ? 'Turn OFF' : 'Turn ON'}
              </button>
              <button 
                className="mode-toggle"
                onClick={() => toggleMode(relay.id)}
              >
                {relay.auto_controlled ? 'Switch to Manual' : 'Switch to Auto'}
              </button>
            </div>
          </div>
        ))}
      </div>
      
      <style jsx>{`
        .relay-control-container {
          padding: 20px;
          background-color: #f5f5f5;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        h2 {
          color: #333;
          margin-bottom: 20px;
          border-bottom: 1px solid #ddd;
          padding-bottom: 10px;
        }
        
        .error-message {
          background-color: #ffebee;
          color: #c62828;
          padding: 10px;
          border-radius: 4px;
          margin-bottom: 15px;
        }
        
        .loading {
          text-align: center;
          padding: 20px;
          color: #666;
        }
        
        .mode-selector {
          margin-bottom: 20px;
          display: flex;
          align-items: center;
        }
        
        .mode-button {
          margin-left: 10px;
          padding: 8px 16px;
          border: 1px solid #ccc;
          background-color: #fff;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.3s;
        }
        
        .mode-button.active {
          background-color: #2196f3;
          color: white;
          border-color: #2196f3;
        }
        
        .relays-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 20px;
        }
        
        .relay-card {
          background-color: white;
          border-radius: 8px;
          padding: 15px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        }
        
        .relay-card h3 {
          margin-top: 0;
          color: #444;
        }
        
        .relay-status, .relay-mode {
          margin-bottom: 10px;
        }
        
        .on {
          color: #4caf50;
          font-weight: bold;
        }
        
        .off {
          color: #f44336;
          font-weight: bold;
        }
        
        .relay-controls {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .toggle-button {
          padding: 8px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-weight: bold;
          transition: background-color 0.3s;
        }
        
        .toggle-button.on {
          background-color: #ffcdd2;
          color: #c62828;
        }
        
        .toggle-button.off {
          background-color: #c8e6c9;
          color: #2e7d32;
        }
        
        .toggle-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .mode-toggle {
          padding: 8px;
          background-color: #e3f2fd;
          border: 1px solid #bbdefb;
          color: #1976d2;
          border-radius: 4px;
          cursor: pointer;
        }
      `}</style>
    </div>
  );
};

export default RelayControl;