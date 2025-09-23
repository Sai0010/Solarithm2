import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// Import components
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import SensorReadings from './components/SensorReadings';
import RelayControl from './components/RelayControl';
import PowerForecast from './components/PowerForecast';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <div className="container">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/sensors" element={<SensorReadings />} />
            <Route path="/relays" element={<RelayControl />} />
            <Route path="/forecast" element={<PowerForecast />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;