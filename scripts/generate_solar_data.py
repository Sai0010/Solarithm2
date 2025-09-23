#!/usr/bin/env python
"""
Synthetic Solar Data Generator
Generates realistic solar sensor data for testing and development.
"""
import os
import sys
import json
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

def load_config():
    """Load hardware configuration"""
    config_path = Path(__file__).parent.parent / "config" / "hardware.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load config from {config_path}")
        return {
            "simulation": {
                "voltage_base": 12.0,
                "voltage_variance": 2.0,
                "current_max": 5.0,
                "temperature_base": 20.0,
                "temperature_variance": 10.0,
                "humidity_base": 50.0,
                "humidity_variance": 20.0,
                "light_max": 1000.0
            }
        }

def calculate_sun_intensity(hour, day_length=12, peak_hour=12):
    """Calculate sun intensity based on hour of day"""
    if hour < 6 or hour > 18:  # Night time
        return 0.0
    
    # Calculate intensity based on sine curve
    hour_normalized = (hour - 6) / day_length
    return max(0, min(1, np.sin(hour_normalized * np.pi)))

def generate_daily_data(config, days=1, interval_minutes=15, start_date=None):
    """Generate synthetic solar data for specified number of days"""
    if start_date is None:
        start_date = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(days=days-1)
    
    # Calculate number of data points
    points_per_day = 24 * 60 // interval_minutes
    total_points = days * points_per_day
    
    # Prepare timestamp array
    timestamps = [
        start_date + datetime.timedelta(minutes=i*interval_minutes)
        for i in range(total_points)
    ]
    
    # Extract simulation parameters
    sim_config = config.get("simulation", {})
    voltage_base = sim_config.get("voltage_base", 12.0)
    voltage_variance = sim_config.get("voltage_variance", 2.0)
    current_max = sim_config.get("current_max", 5.0)
    temp_base = sim_config.get("temperature_base", 20.0)
    temp_variance = sim_config.get("temperature_variance", 10.0)
    humidity_base = sim_config.get("humidity_base", 50.0)
    humidity_variance = sim_config.get("humidity_variance", 20.0)
    light_max = sim_config.get("light_max", 1000.0)
    
    # Generate data
    data = []
    for ts in timestamps:
        hour = ts.hour + ts.minute/60
        sun_intensity = calculate_sun_intensity(hour)
        
        # Add some day-to-day variation
        daily_factor = 0.8 + 0.4 * random.random()  # 0.8 to 1.2
        
        # Calculate sensor values
        voltage = voltage_base + (random.random() - 0.5) * voltage_variance
        current = sun_intensity * current_max * daily_factor
        power = voltage * current
        
        # Temperature increases with sun intensity
        temperature = temp_base + sun_intensity * temp_variance * daily_factor
        
        # Humidity tends to be lower when temperature is higher
        humidity = humidity_base - sun_intensity * 20 + random.random() * humidity_variance
        humidity = max(10, min(90, humidity))  # Clamp between 10% and 90%
        
        # Light level directly related to sun intensity
        light_level = sun_intensity * light_max * daily_factor
        
        data.append({
            "timestamp": ts.isoformat(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "power": round(power, 2),
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "light_level": round(light_level, 0)
        })
    
    return data

def save_to_csv(data, output_path):
    """Save generated data to CSV file"""
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return output_path

def upload_to_api(data, api_url):
    """Upload generated data to API"""
    if not API_AVAILABLE:
        print("Warning: requests module not available, skipping API upload")
        return False
    
    try:
        # Split into batches of 100 records to avoid large payloads
        batch_size = 100
        success = True
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            response = requests.post(
                f"{api_url}/sensors/batch", 
                json={"readings": batch}
            )
            
            if response.status_code != 200:
                print(f"Error uploading batch {i//batch_size}: {response.status_code}")
                print(response.text)
                success = False
        
        if success:
            print(f"Successfully uploaded {len(data)} records to {api_url}")
        return success
    
    except Exception as e:
        print(f"Error uploading to API: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic solar sensor data")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data to generate")
    parser.add_argument("--interval", type=int, default=15, help="Data interval in minutes")
    parser.add_argument("--output", type=str, default="../ml_model/data/synthetic_data.csv", 
                        help="Output CSV file path")
    parser.add_argument("--api", type=str, help="API URL to upload data")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Generate data
    print(f"Generating {args.days} days of data at {args.interval}-minute intervals...")
    data = generate_daily_data(config, days=args.days, interval_minutes=args.interval)
    
    # Save to CSV
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output))
    save_to_csv(data, output_path)
    
    # Upload to API if specified
    if args.api:
        upload_to_api(data, args.api)

if __name__ == "__main__":
    main()