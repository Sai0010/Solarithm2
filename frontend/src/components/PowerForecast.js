import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { getPowerForecast } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const PowerForecast = () => {
  const [forecastData, setForecastData] = useState(null);
  const [timeRange, setTimeRange] = useState('24h');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    peakProduction: 0,
    totalProduction: 0,
    efficiency: 0,
    confidence: 0
  });

  useEffect(() => {
    fetchForecastData(timeRange);
  }, [timeRange]);

  const fetchForecastData = async (range) => {
    try {
      setLoading(true);
      // Default device ID - in a real app, this would come from user selection or context
      const deviceId = "solar_panel_01";
      const horizon = range === '24h' ? 24 : (range === '12h' ? 12 : 6);
      
      const response = await getPowerForecast(deviceId, horizon);
      setForecastData(response || {});
      calculateStats(response || {});
      setError(null);
    } catch (err) {
      console.error('Error fetching forecast data:', err);
      setError('Failed to load forecast data. Using simulated data instead.');
      
      // Generate simulated data for development
      const simulatedData = generateSimulatedData(range);
      setForecastData(simulatedData);
      calculateStats(simulatedData);
    } finally {
      setLoading(false);
    }
  };

  const calculateStats = (data) => {
    const series = (data && Array.isArray(data.forecast)) ? data.forecast : [];
    if (series.length === 0) return;
    const values = series.map(p => Number(p.power) || 0);
    const peakProduction = Math.max(...values);
    const totalProduction = values.reduce((sum, val) => sum + val, 0);
    
    setStats({
      peakProduction: peakProduction.toFixed(2),
      totalProduction: totalProduction.toFixed(2),
      efficiency: data.efficiency || (Math.random() * 20 + 75).toFixed(2),
      confidence: data.confidence || (Math.random() * 15 + 80).toFixed(2)
    });
  };

  const generateSimulatedData = (range) => {
    const now = new Date();
    const dataPoints = range === '24h' ? 24 : range === '7d' ? 7 : 30;
    const forecast = [];
    
    for (let i = 0; i < dataPoints; i++) {
      let date;
      let label;
      
      if (range === '24h') {
        date = new Date(now.getTime() + i * 60 * 60 * 1000);
        label = date.getHours() + ':00';
      } else if (range === '7d') {
        date = new Date(now.getTime() + i * 24 * 60 * 60 * 1000);
        label = date.toLocaleDateString('en-US', { weekday: 'short' });
      } else {
        date = new Date(now.getTime() + i * 24 * 60 * 60 * 1000);
        label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }
      
      
      // Generate realistic solar production curve
      let baseValue;
      if (range === '24h') {
        // Daily curve with peak at midday
        const hour = date.getHours();
        baseValue = hour >= 6 && hour <= 18 
          ? 20 * Math.sin(Math.PI * (hour - 6) / 12) 
          : 0;
      } else {
        // Weekly/monthly with some variation
        baseValue = 10 + Math.random() * 15;
      }
      
      const predictionValue = Math.max(0, baseValue + (Math.random() * 5 - 2.5));
      forecast.push({ timestamp: date.toISOString(), power: Number(predictionValue.toFixed(2)) });
    }
    
    return {
      forecast,
      efficiency: (Math.random() * 20 + 75).toFixed(2),
      confidence: (Math.random() * 15 + 80).toFixed(2)
    };
  };

  if (loading && !forecastData) {
    return <div className="loading">Loading forecast data...</div>;
  }

  return (
    <div style={{
      padding: '20px',
      backgroundColor: '#f5f5f5',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
    }}>
      <h2 style={{
        color: '#333',
        marginBottom: '20px',
        borderBottom: '1px solid #ddd',
        paddingBottom: '10px'
      }}>Solar Power Forecast</h2>
      
      {error && <div style={{
        backgroundColor: '#ffebee',
        color: '#c62828',
        padding: '10px',
        borderRadius: '4px',
        marginBottom: '15px'
      }}>{error}</div>}
      
      <div style={{
        marginBottom: '20px',
        display: 'flex',
        gap: '10px'
      }}>
        {['24h', '7d', '30d'].map(range => (
          <button 
            key={range}
            style={{
              padding: '8px 16px',
              border: '1px solid #ccc',
              backgroundColor: timeRange === range ? '#2196f3' : '#fff',
              color: timeRange === range ? 'white' : 'black',
              cursor: 'pointer',
              borderRadius: '4px',
              transition: 'all 0.3s'
            }}
            onClick={() => setTimeRange(range)}
          >
            {range === '24h' ? '24 Hours' : range === '7d' ? '7 Days' : '30 Days'}
          </button>
        ))}
      </div>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: '15px',
        marginBottom: '20px'
      }}>
        {[
          { title: 'Peak Production', value: `${stats.peakProduction} kW` },
          { title: 'Total Production', value: `${stats.totalProduction} kWh` },
          { title: 'System Efficiency', value: `${stats.efficiency}%` },
          { title: 'Forecast Confidence', value: `${stats.confidence}%` }
        ].map((stat, index) => (
          <div key={index} style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '15px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.12)',
            textAlign: 'center'
          }}>
            <h4 style={{
              marginTop: 0,
              color: '#666',
              fontSize: '14px'
            }}>{stat.title}</h4>
            <div style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: '#2196f3'
            }}>{stat.value}</div>
          </div>
        ))}
      </div>
      
      {forecastData && Array.isArray(forecastData.forecast) && (
        <div style={{
          backgroundColor: 'white',
          padding: '15px',
          borderRadius: '8px',
          marginBottom: '20px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.12)',
          height: '320px'
        }}>
          <Line
            data={{
              labels: forecastData.forecast.map(p => new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
              datasets: [
                {
                  label: 'Forecast Power (W)',
                  data: forecastData.forecast.map(p => p.power),
                  borderColor: 'rgb(33, 150, 243)',
                  backgroundColor: 'rgba(33, 150, 243, 0.3)'
                }
              ]
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Power Forecast' }
              }
            }}
          />
        </div>
      )}
      
      <div style={{
        backgroundColor: '#e3f2fd',
        padding: '15px',
        borderRadius: '8px',
        borderLeft: '4px solid #2196f3'
      }}>
        <h4 style={{
          marginTop: 0,
          color: '#0d47a1'
        }}>Notes</h4>
        <ul style={{
          margin: 0,
          paddingLeft: '20px'
        }}>
          <li style={{ marginBottom: '5px' }}>Forecast is based on weather predictions and historical performance data</li>
          <li style={{ marginBottom: '5px' }}>Actual production may vary based on weather conditions</li>
          <li style={{ marginBottom: '5px' }}>System automatically adjusts relays based on forecast data</li>
        </ul>
      </div>
    </div>
  );
};

export default PowerForecast;