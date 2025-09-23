# SolArithm Hardware-to-Code Mapping

This document provides a comprehensive mapping between hardware components, GPIO pins, scripts, API endpoints, and database columns in the SolArithm system.

## Hardware Components Overview

| Component | Description | Purpose |
|-----------|-------------|---------|
| BH1750 | Light intensity sensor | Measures solar irradiance (lux) |
| ACS712 | Current sensor | Measures solar panel current output (A) |
| ZMPT101B | Voltage sensor | Measures solar panel voltage (V) |
| DHT22 | Temperature & humidity sensor | Measures ambient conditions |
| Voltage Divider | Resistor network | Scales down voltage for ADC input |
| Relay Module | 4-channel relay board | Controls power flow between components |
| MCP3008 | 8-channel ADC | Converts analog sensor readings to digital |
| Charge Controller | Power management | Manages battery charging/discharging |

## Hardware-to-Code Mapping Table

| Component | GPIO/MCP3008 | Script | Function | API Endpoint | DB Column |
|-----------|--------------|--------|----------|--------------|-----------|
| BH1750 | I2C (GPIO 2,3) | sensor_reader.py | read_light_sensor() | POST /readings/ | lux |
| ACS712 | MCP3008 CH0 | sensor_reader.py | read_current_sensor() | POST /readings/ | current_a |
| ZMPT101B | MCP3008 CH1 | sensor_reader.py | read_voltage_sensor() | POST /readings/ | voltage_v |
| DHT22 (Temp) | GPIO 4 | sensor_reader.py | read_dht_sensor() | POST /readings/ | temp_c |
| DHT22 (Humidity) | GPIO 4 | sensor_reader.py | read_dht_sensor() | POST /readings/ | humidity_pct |
| Relay 1 (Solar Panel) | GPIO 17 | relay_controller.py | set_relay_state() | POST /relays/set | relay_1 |
| Relay 2 (Battery) | GPIO 27 | relay_controller.py | set_relay_state() | POST /relays/set | relay_2 |
| Relay 3 (Load) | GPIO 22 | relay_controller.py | set_relay_state() | POST /relays/set | relay_3 |
| Relay 4 (Auxiliary) | GPIO 23 | relay_controller.py | set_relay_state() | POST /relays/set | relay_4 |

## Data Flow

```
Hardware Sensors → sensor_reader.py → API (/readings/) → Database (sensor_readings table)
API (/relays/set) → relay_controller.py → Relay Hardware → Physical Components
```

## Wiring Diagram Reference

For detailed wiring instructions, refer to the following connections:

1. **MCP3008 ADC:**
   - VDD → 3.3V
   - VREF → 3.3V
   - AGND → GND
   - CLK → GPIO 11 (SCLK)
   - DOUT → GPIO 9 (MISO)
   - DIN → GPIO 10 (MOSI)
   - CS → GPIO 8 (CE0)
   - DGND → GND

2. **BH1750 Light Sensor:**
   - VCC → 3.3V
   - GND → GND
   - SCL → GPIO 3 (SCL)
   - SDA → GPIO 2 (SDA)

3. **DHT22 Temperature/Humidity:**
   - VCC → 3.3V
   - DATA → GPIO 4
   - GND → GND

4. **Relay Module:**
   - VCC → 5V
   - GND → GND
   - IN1 → GPIO 17
   - IN2 → GPIO 27
   - IN3 → GPIO 22
   - IN4 → GPIO 23

## Configuration

All pin assignments and hardware settings are stored in `config/hardware.json`. The sensor and relay scripts dynamically load these settings at runtime.

## Validation

The system performs validation checks on startup to ensure:
- All required hardware components have valid pin assignments
- No pin conflicts exist between components
- All components map to appropriate database columns

If validation fails, detailed error messages are logged to assist in troubleshooting.