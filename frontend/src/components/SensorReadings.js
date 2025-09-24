import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import axios from 'axios';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const SensorReadings = () => {
  const [currentReadings, setCurrentReadings] = useState({
    voltage: 0,
    current: 0,
    power: 0,
    temperature: 0,
    humidity: 0,
    lightLevel: 0,
  });
  
  const [historicalData, setHistoricalData] = useState({
    labels: [],
    datasets: [
      {
        label: 'Power (W)',
        data: [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'Light Level (lux)',
        data: [],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  });

  // Fetch current sensor readings
  const fetchCurrentReadings = async () => {
    try {
      const response = await axios.get('/api/latest', {
        params: { device_id: 'solar_panel_01' }
      });
      setCurrentReadings(response.data);
    } catch (error) {
      console.error('Error fetching current readings:', error);
      // Use simulated data if API fails
      simulateCurrentReadings();
    }
  };

  // Fetch historical sensor data
  const fetchHistoricalData = async () => {
    try {
      // Since there's no direct historical endpoint, we'll simulate it for now
      // In a real implementation, you would create a backend endpoint for this
      simulateHistoricalData();
      
    } catch (error) {
      console.error('Error fetching historical data:', error);
      // Use simulated data if API fails
      simulateHistoricalData();
    }
  };

  // Simulate current readings for development/testing
  const simulateCurrentReadings = () => {
    const time = new Date().getHours();
    const sunIntensity = time > 6 && time < 18 ? Math.sin((time - 6) * Math.PI / 12) : 0;
    
    setCurrentReadings({
      voltage: (12 + Math.random() * 2).toFixed(1),
      current: (sunIntensity * 5 + Math.random()).toFixed(2),
      power: (sunIntensity * 60 + Math.random() * 10).toFixed(1),
      temperature: (20 + sunIntensity * 10 + Math.random() * 2).toFixed(1),
      humidity: (50 + Math.random() * 20).toFixed(1),
      lightLevel: (sunIntensity * 1000 + Math.random() * 100).toFixed(0),
    });
  };

  // Simulate historical data for development/testing
  const simulateHistoricalData = () => {
    const currentHour = new Date().getHours();
    const labels = [];
    const powerData = [];
    const lightData = [];
    
    // Generate data for the past 24 hours
    for (let i = 0; i < 24; i++) {
      const hour = (currentHour - 23 + i + 24) % 24;
      const label = `${hour}:00`;
      labels.push(label);
      
      const sunIntensity = hour > 6 && hour < 18 ? Math.sin((hour - 6) * Math.PI / 12) : 0;
      powerData.push((sunIntensity * 60 + Math.random() * 10).toFixed(1));
      lightData.push((sunIntensity * 1000 + Math.random() * 100).toFixed(0));
    }
    
    setHistoricalData({
      labels,
      datasets: [
        {
          label: 'Power (W)',
          data: powerData,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
        {
          label: 'Light Level (lux)',
          data: lightData,
          borderColor: 'rgb(53, 162, 235)',
          backgroundColor: 'rgba(53, 162, 235, 0.5)',
        },
      ],
    });
  };

  // Initial data load and set up auto-refresh
  useEffect(() => {
    fetchCurrentReadings();
    fetchHistoricalData();
    
    // Set up auto-refresh every 30 seconds
    const intervalId = setInterval(() => {
      fetchCurrentReadings();
      fetchHistoricalData();
    }, 30000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="sensor-readings">
      <h2>Current Sensor Readings</h2>
      <div className="readings-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
        <div className="reading-card" style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3>Voltage</h3>
          <p className="reading-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentReadings.voltage} V</p>
        </div>
        <div className="reading-card" style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3>Current</h3>
          <p className="reading-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentReadings.current} A</p>
        </div>
        <div className="reading-card" style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3>Power</h3>
          <p className="reading-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentReadings.power} W</p>
        </div>
        <div className="reading-card" style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3>Temperature</h3>
          <p className="reading-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentReadings.temperature} °C</p>
        </div>
        <div className="reading-card" style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3>Humidity</h3>
          <p className="reading-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentReadings.humidity} %</p>
        </div>
        <div className="reading-card" style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3>Light Level</h3>
          <p className="reading-value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentReadings.lightLevel} lux</p>
        </div>
      </div>
      
      <h2 style={{ marginTop: '2rem' }}>Historical Data</h2>
      <div className="chart-container" style={{ marginTop: '1rem', height: '400px' }}>
        <Line 
          data={historicalData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'top',
              },
              title: {
                display: true,
                text: '24-Hour Sensor History',
              },
            },
          }}
        />
      </div>
    </div>
  );
};

export default SensorReadings;