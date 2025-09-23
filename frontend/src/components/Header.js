import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="App-header">
      <div className="logo">
        <h1>SolArithm</h1>
      </div>
      <nav>
        <ul className="nav-links">
          <li><Link to="/">Dashboard</Link></li>
          <li><Link to="/sensors">Sensor Data</Link></li>
          <li><Link to="/relays">Relay Control</Link></li>
          <li><Link to="/forecast">Power Forecast</Link></li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;