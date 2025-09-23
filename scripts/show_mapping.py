#!/usr/bin/env python3
"""
SolArithm Hardware Mapping Tool

This script displays the mapping between hardware components, GPIO pins,
scripts, API endpoints, and database columns to help developers understand
the system architecture.
"""

import os
import sys
import json
import argparse
from tabulate import tabulate
from colorama import init, Fore, Style

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize colorama for cross-platform colored terminal output
init()

def load_config():
    """Load hardware configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'hardware.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Configuration file not found at {config_path}{Style.RESET_ALL}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"{Fore.RED}Error: Invalid JSON in configuration file{Style.RESET_ALL}")
        sys.exit(1)

def validate_config(config):
    """Validate hardware configuration for completeness and correctness."""
    errors = []
    
    # Check for required sections
    required_sections = ['sensors', 'relays', 'adc']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    if errors:
        return False, errors
    
    # Check sensor configurations
    for sensor_name, sensor_config in config['sensors'].items():
        if 'type' not in sensor_config:
            errors.append(f"Sensor {sensor_name} missing 'type' field")
        
        if sensor_config.get('type') == 'analog':
            if 'adc' not in sensor_config:
                errors.append(f"Analog sensor {sensor_name} missing 'adc' field")
            if 'channel' not in sensor_config:
                errors.append(f"Analog sensor {sensor_name} missing 'channel' field")
        
        if sensor_config.get('type') == 'digital':
            if 'pin' not in sensor_config:
                errors.append(f"Digital sensor {sensor_name} missing 'pin' field")
    
    # Check relay configurations
    for relay_name, relay_config in config['relays'].items():
        if 'pin' not in relay_config:
            errors.append(f"Relay {relay_name} missing 'pin' field")
        if 'db_column' not in relay_config:
            errors.append(f"Relay {relay_name} missing 'db_column' field")
    
    # Check for pin conflicts
    used_pins = {}
    
    # Check sensor pins
    for sensor_name, sensor_config in config['sensors'].items():
        if sensor_config.get('type') == 'digital':
            pin = sensor_config.get('pin')
            if pin in used_pins:
                errors.append(f"Pin conflict: {pin} used by both {sensor_name} and {used_pins[pin]}")
            else:
                used_pins[pin] = sensor_name
        
        if sensor_config.get('type') == 'i2c':
            for pin_type in ['scl_pin', 'sda_pin']:
                if pin_type in sensor_config:
                    pin = sensor_config[pin_type]
                    # I2C pins can be shared, so no conflict check needed
                    used_pins[pin] = f"{sensor_name} ({pin_type})"
    
    # Check relay pins
    for relay_name, relay_config in config['relays'].items():
        pin = relay_config.get('pin')
        if pin in used_pins and not used_pins[pin].startswith(relay_name):
            errors.append(f"Pin conflict: {pin} used by both {relay_name} and {used_pins[pin]}")
        else:
            used_pins[pin] = relay_name
    
    return len(errors) == 0, errors

def generate_sensor_table(config):
    """Generate a table showing sensor mappings."""
    headers = ["Sensor", "Type", "GPIO/Channel", "Script", "Function", "API Endpoint", "DB Column"]
    rows = []
    
    for sensor_name, sensor_config in config['sensors'].items():
        sensor_type = sensor_config.get('type', 'unknown')
        
        if sensor_type == 'analog':
            gpio_channel = f"{sensor_config.get('adc', 'unknown')} CH{sensor_config.get('channel', '?')}"
            function_name = f"read_{sensor_name}_sensor()"
        elif sensor_type == 'digital':
            gpio_channel = f"GPIO {sensor_config.get('pin', '?')}"
            function_name = f"read_{sensor_name}_sensor()"
        elif sensor_type == 'i2c':
            gpio_channel = f"I2C (GPIO {sensor_config.get('scl_pin', '?')},{sensor_config.get('sda_pin', '?')})"
            function_name = f"read_{sensor_name}_sensor()"
        else:
            gpio_channel = "Unknown"
            function_name = "unknown()"
        
        # Handle special case for DHT22 which has multiple readings
        if sensor_name == 'dht22' and 'db_columns' in sensor_config:
            for reading_type, db_column in sensor_config['db_columns'].items():
                rows.append([
                    f"{sensor_name.upper()} ({reading_type})",
                    sensor_type,
                    gpio_channel,
                    "sensor_reader.py",
                    function_name,
                    "POST /readings/",
                    db_column
                ])
        else:
            db_column = sensor_config.get('db_column', 'unknown')
            rows.append([
                sensor_name.upper(),
                sensor_type,
                gpio_channel,
                "sensor_reader.py",
                function_name,
                "POST /readings/",
                db_column
            ])
    
    return headers, rows

def generate_relay_table(config):
    """Generate a table showing relay mappings."""
    headers = ["Relay", "GPIO Pin", "Script", "Function", "API Endpoint", "DB Column"]
    rows = []
    
    for relay_name, relay_config in config['relays'].items():
        rows.append([
            relay_config.get('name', relay_name),
            f"GPIO {relay_config.get('pin', '?')}",
            "relay_controller.py",
            "set_relay_state()",
            "POST /relays/set",
            relay_config.get('db_column', 'unknown')
        ])
    
    return headers, rows

def print_mapping():
    """Print the hardware-to-code mapping tables."""
    config = load_config()
    is_valid, errors = validate_config(config)
    
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}SolArithm Hardware-to-Code Mapping{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")
    
    if not is_valid:
        print(f"{Fore.RED}Configuration validation failed with the following errors:{Style.RESET_ALL}")
        for error in errors:
            print(f"{Fore.RED} - {error}{Style.RESET_ALL}")
        print("\nDisplaying available mapping information anyway:\n")
    
    # Print sensor table
    sensor_headers, sensor_rows = generate_sensor_table(config)
    print(f"{Fore.GREEN}Sensor Mapping:{Style.RESET_ALL}")
    print(tabulate(sensor_rows, headers=sensor_headers, tablefmt="grid"))
    print()
    
    # Print relay table
    relay_headers, relay_rows = generate_relay_table(config)
    print(f"{Fore.GREEN}Relay Mapping:{Style.RESET_ALL}")
    print(tabulate(relay_rows, headers=relay_headers, tablefmt="grid"))
    print()
    
    # Print data flow
    print(f"{Fore.GREEN}Data Flow:{Style.RESET_ALL}")
    print("Hardware Sensors → sensor_reader.py → API (/readings/) → Database (sensor_readings table)")
    print("API (/relays/set) → relay_controller.py → Relay Hardware → Physical Components")
    print()
    
    # Print validation status
    if is_valid:
        print(f"{Fore.GREEN}✓ Configuration validation passed. All components properly mapped.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Configuration validation failed. See errors above.{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

def main():
    """Main function to parse arguments and display mapping."""
    parser = argparse.ArgumentParser(description='Display hardware-to-code mapping for SolArithm')
    parser.add_argument('--validate-only', action='store_true', help='Only validate the configuration without displaying tables')
    args = parser.parse_args()
    
    if args.validate_only:
        config = load_config()
        is_valid, errors = validate_config(config)
        if is_valid:
            print(f"{Fore.GREEN}✓ Configuration validation passed. All components properly mapped.{Style.RESET_ALL}")
            sys.exit(0)
        else:
            print(f"{Fore.RED}✗ Configuration validation failed with the following errors:{Style.RESET_ALL}")
            for error in errors:
                print(f"{Fore.RED} - {error}{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print_mapping()

if __name__ == "__main__":
    main()