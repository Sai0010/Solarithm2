# SolArithm Raspberry Pi Scripts

This directory contains scripts for running SolArithm on Raspberry Pi devices to collect sensor data and control relays.

## Scripts Overview

- `sensor_reader.py`: Reads data from connected sensors and sends to the SolArithm API
- `relay_controller.py`: Controls relays based on commands from the SolArithm API

## Hardware Requirements

- Raspberry Pi (3B+ or 4 recommended)
- DHT22 temperature/humidity sensor
- MCP3008 analog-to-digital converter
- Voltage divider for solar panel voltage measurement
- Current sensor (e.g., ACS712)
- Light sensor (photoresistor or dedicated lux sensor)
- Relay module (4-channel recommended)

## Installation

1. Install required Python packages:

```bash
pip install RPi.GPIO adafruit-blinka adafruit-circuitpython-dht adafruit-circuitpython-mcp3xxx requests
```

2. Connect hardware according to the following pin configuration:
   - DHT22: GPIO4 (default, configurable)
   - MCP3008: 
     - SPI connections (MOSI, MISO, SCLK)
     - CS: GPIO5
   - Relays: GPIO17, GPIO18, GPIO27, GPIO22 (default, configurable)

## Usage

### Sensor Reader

```bash
python sensor_reader.py --api-url http://your-server:8000/api/sensors --interval 60
```

Options:
- `--device-id`: Device identifier (default: solar_panel_01)
- `--api-url`: API endpoint URL (default: http://localhost:8000/api/sensors)
- `--interval`: Reading interval in seconds (default: 60)
- `--simulate`: Run in simulation mode without hardware
- `--dht-pin`: GPIO pin for DHT22 sensor (default: 4)

### Relay Controller

```bash
python relay_controller.py --api-url http://your-server:8000/api/relays/status --interval 30
```

Options:
- `--device-id`: Device identifier (default: solar_panel_01)
- `--api-url`: API endpoint URL (default: http://localhost:8000/api/relays/status)
- `--interval`: Poll interval in seconds (default: 30)
- `--simulate`: Run in simulation mode without hardware
- `--relay-pins`: Comma-separated list of GPIO pins for relays (default: 17,18,27,22)

## Simulation Mode

Both scripts support a simulation mode that doesn't require actual hardware. This is useful for:
- Development on non-Raspberry Pi systems
- Testing without complete hardware setup
- Demonstration purposes

To run in simulation mode, add the `--simulate` flag:

```bash
python sensor_reader.py --simulate
python relay_controller.py --simulate
```

## Automatic Startup

To run these scripts automatically at boot:

1. Create a systemd service file for each script:

```bash
sudo nano /etc/systemd/system/solarithm-sensor.service
```

2. Add the following content (adjust paths as needed):

```
[Unit]
Description=SolArithm Sensor Reader
After=network.target

[Service]
ExecStart=/usr/bin/python3 /path/to/sensor_reader.py --api-url http://your-server:8000/api/sensors
WorkingDirectory=/path/to/scripts
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

3. Create a similar file for the relay controller:

```bash
sudo nano /etc/systemd/system/solarithm-relay.service
```

4. Enable and start the services:

```bash
sudo systemctl enable solarithm-sensor.service
sudo systemctl start solarithm-sensor.service
sudo systemctl enable solarithm-relay.service
sudo systemctl start solarithm-relay.service
```

## Troubleshooting

- Check logs with `journalctl -u solarithm-sensor.service` or `journalctl -u solarithm-relay.service`
- Ensure proper permissions for GPIO access (run as root or add user to gpio group)
- Verify hardware connections and pin configurations
- Test API connectivity with `curl` or similar tools