#!/usr/bin/env python3
"""
Manufacturing Time-Series Data Generator
Generates realistic manufacturing data for ML Engineer technical assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import csv

class ManufacturingDataGenerator:
    def __init__(self):
        # Fault codes and their characteristics
        self.fault_codes = {
            "0": {"name": "No Fault", "probability": 0.85, "duration_range": (0, 0), "speed_impact": 1.0},
            "E001": {"name": "Emergency Stop", "probability": 0.02, "duration_range": (15, 45), "speed_impact": 0.0},
            "E002": {"name": "Material Jam", "probability": 0.04, "duration_range": (5, 20), "speed_impact": 0.0},
            "E003": {"name": "Temperature High", "probability": 0.03, "duration_range": (10, 30), "speed_impact": 0.3},
            "E004": {"name": "Vibration Exceeded", "probability": 0.03, "duration_range": (5, 15), "speed_impact": 0.5},
            "E005": {"name": "Feeder Empty", "probability": 0.02, "duration_range": (8, 25), "speed_impact": 0.0},
            "E006": {"name": "Quality Issue", "probability": 0.01, "duration_range": (10, 20), "speed_impact": 0.7}
        }
        
        # Machine configurations
        self.machines = {
            "MC001": {"max_speed": 120, "normal_speed": 100, "temp_baseline": 65},
            "MC002": {"max_speed": 150, "normal_speed": 130, "temp_baseline": 70},
            "MC003": {"max_speed": 110, "normal_speed": 95, "temp_baseline": 60}
        }
        
        # Shift patterns (speed variations throughout the day)
        self.shift_patterns = {
            "morning": {"start": 6, "end": 14, "efficiency": 0.95},
            "afternoon": {"start": 14, "end": 22, "efficiency": 0.90},
            "night": {"start": 22, "end": 6, "efficiency": 0.85}
        }

    def get_shift_efficiency(self, hour):
        """Get efficiency multiplier based on time of day"""
        if 6 <= hour < 14:
            return self.shift_patterns["morning"]["efficiency"]
        elif 14 <= hour < 22:
            return self.shift_patterns["afternoon"]["efficiency"]
        else:
            return self.shift_patterns["night"]["efficiency"]

    def generate_fault_sequence(self, duration_minutes):
        """Generate realistic fault sequence with correlated events"""
        fault_sequence = []
        current_time = 0
        current_fault = "0"
        fault_end_time = 0
        
        while current_time < duration_minutes:
            # Check if current fault should end
            if current_time >= fault_end_time and current_fault != "0":
                current_fault = "0"
                fault_end_time = 0
            
            # Determine if new fault should start (only if no current fault)
            if current_fault == "0":
                # Higher probability of faults during shift changes
                hour = (current_time // 60) % 24
                fault_multiplier = 1.5 if hour in [6, 14, 22] else 1.0
                
                for fault_code, properties in self.fault_codes.items():
                    if fault_code == "0":
                        continue
                    
                    if random.random() < (properties["probability"] * fault_multiplier * 0.1):  # 0.1 per 5-min interval
                        current_fault = fault_code
                        duration = random.randint(*properties["duration_range"])
                        fault_end_time = current_time + duration
                        break
            
            fault_sequence.append(current_fault)
            current_time += 5  # 5-minute intervals
            
        return fault_sequence

    def calculate_oee(self, speed, fault_code, target_speed, quality_factor=0.98):
        """Calculate OEE based on availability, performance, and quality"""
        # Availability (machine running without faults)
        availability = 1.0 if fault_code == "0" else 0.0
        
        # Performance (actual speed vs target speed)
        performance = min(speed / target_speed, 1.0) if target_speed > 0 else 0.0
        
        # Quality (assume some quality losses during certain conditions)
        quality = quality_factor
        if fault_code in ["E003", "E004"]:  # Temperature/vibration issues affect quality
            quality *= 0.9
        
        oee = availability * performance * quality * 100
        return round(oee, 1)

    def generate_machine_data(self, machine_id, start_date, days=7, interval_minutes=5):
        """Generate time-series data for a single machine"""
        
        machine_config = self.machines[machine_id]
        total_intervals = (days * 24 * 60) // interval_minutes
        
        # Generate timestamps
        timestamps = []
        current_time = start_date
        for _ in range(total_intervals):
            timestamps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        # Generate fault sequence
        fault_sequence = self.generate_fault_sequence(days * 24 * 60)
        
        # Initialize data lists
        data = []
        counter = random.randint(10000, 50000)  # Starting counter value
        
        for i, timestamp in enumerate(timestamps):
            fault_code = fault_sequence[i] if i < len(fault_sequence) else "0"
            fault_properties = self.fault_codes[fault_code]
            
            # Calculate base speed with shift efficiency
            hour = timestamp.hour
            shift_efficiency = self.get_shift_efficiency(hour)
            base_speed = machine_config["normal_speed"] * shift_efficiency
            
            # Apply fault impact
            if fault_code == "0":
                # Normal operation with some random variation
                speed = base_speed * random.uniform(0.90, 1.05)
                feeder_status = "RUNNING"
            else:
                speed = base_speed * fault_properties["speed_impact"]
                feeder_status = "STOPPED" if speed == 0 else "FAULTED"
            
            # Round speed and ensure non-negative
            speed = max(0, round(speed, 1))
            
            # Update counter (only when running)
            if speed > 0:
                boards_produced = (speed * interval_minutes)
                counter += int(boards_produced)
            
            # Temperature calculation
            temp_baseline = machine_config["temp_baseline"]
            if fault_code == "E003":  # High temperature fault
                temperature = temp_baseline + random.uniform(15, 25)
            elif speed > 0:
                # Temperature correlates with speed
                temp_factor = speed / machine_config["normal_speed"]
                temperature = temp_baseline + (temp_factor * random.uniform(5, 15)) + random.gauss(0, 2)
            else:
                # Machine cooling down when stopped
                temperature = temp_baseline - random.uniform(5, 15)
            
            temperature = round(temperature, 1)
            
            # Vibration calculation
            if fault_code == "E004":  # Vibration fault
                vibration = random.uniform(8, 15)
            elif speed > 0:
                # Normal vibration correlates with speed
                base_vibration = (speed / machine_config["max_speed"]) * 3
                vibration = base_vibration + random.gauss(0, 0.5)
            else:
                vibration = random.uniform(0, 0.2)
            
            vibration = max(0, round(vibration, 1))
            
            # Calculate OEE
            oee = self.calculate_oee(speed, fault_code, machine_config["normal_speed"])
            
            # Create data row
            row = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": machine_id,
                "speed_board_per_min": speed,
                "fault_code": fault_code,
                "counter_total": counter,
                "feeder_status": feeder_status,
                "temperature_c": temperature,
                "vibration_mm_s": vibration,
                "oee_percent": oee
            }
            
            data.append(row)
        
        return data

    def generate_dataset(self, start_date_str="2024-01-15", days=7, output_file="machine_data.csv"):
        """Generate complete dataset for all machines"""
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        all_data = []
        
        print(f"Generating {days} days of manufacturing data...")
        
        # Generate data for each machine
        for machine_id in self.machines.keys():
            print(f"Processing {machine_id}...")
            machine_data = self.generate_machine_data(machine_id, start_date, days)
            all_data.extend(machine_data)
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['timestamp', 'machine_id']).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Generate summary statistics
        self.generate_summary_report(df, output_file.replace('.csv', '_summary.txt'))
        
        print(f"\nDataset generated successfully!")
        print(f"File: {output_file}")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Machines: {', '.join(df['machine_id'].unique())}")
        
        return df

    def generate_summary_report(self, df, summary_file):
        """Generate a summary report of the dataset"""
        
        with open(summary_file, 'w') as f:
            f.write("MANUFACTURING DATASET SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset Overview:\n")
            f.write(f"- Total Records: {len(df):,}\n")
            f.write(f"- Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            f.write(f"- Machines: {', '.join(sorted(df['machine_id'].unique()))}\n")
            f.write(f"- Data Interval: 5 minutes\n\n")
            
            # Fault analysis
            f.write("Fault Code Distribution:\n")
            fault_counts = df['fault_code'].value_counts()
            for fault, count in fault_counts.items():
                fault_name = self.fault_codes[fault]["name"]
                percentage = (count / len(df)) * 100
                f.write(f"- {fault} ({fault_name}): {count:,} records ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Machine performance
            f.write("Machine Performance Summary:\n")
            for machine in sorted(df['machine_id'].unique()):
                machine_data = df[df['machine_id'] == machine]
                avg_speed = machine_data['speed_board_per_min'].mean()
                avg_oee = machine_data['oee_percent'].mean()
                uptime = (machine_data['fault_code'] == '0').mean() * 100
                
                f.write(f"- {machine}:\n")
                f.write(f"  * Average Speed: {avg_speed:.1f} boards/min\n")
                f.write(f"  * Average OEE: {avg_oee:.1f}%\n")
                f.write(f"  * Uptime: {uptime:.1f}%\n")
            f.write("\n")
            
            # Operational insights
            f.write("Key Insights for ML Model Development:\n")
            f.write("- Dataset includes realistic fault patterns with varying durations\n")
            f.write("- Temperature and vibration correlate with machine performance\n")
            f.write("- Shift patterns affect overall efficiency\n")
            f.write("- Counter values show cumulative production tracking\n")
            f.write("- OEE calculations based on availability, performance, and quality\n")

    def generate_fault_dictionary(self, output_file="fault_codes.json"):
        """Generate fault code dictionary for reference"""
        import json
        
        fault_dict = {}
        for code, properties in self.fault_codes.items():
            fault_dict[code] = properties["name"]
        
        with open(output_file, 'w') as f:
            json.dump(fault_dict, f, indent=2)
        
        print(f"Fault dictionary saved to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = ManufacturingDataGenerator()
    
    # Generate 7 days of data (2016 records per machine, 6048 total)
    df = generator.generate_dataset(
        start_date_str="2024-01-15",
        days=7,
        output_file="machine_data.csv"
    )
    
    # Generate fault dictionary
    generator.generate_fault_dictionary()
    
    # Display sample data
    print("\nSample Data Preview:")
    print(df.head(10).to_string(index=False))
    
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Show fault distribution
    print("\nFault Code Distribution:")
    fault_dist = df['fault_code'].value_counts()
    for fault, count in fault_dist.items():
        fault_name = generator.fault_codes[fault]["name"]
        print(f"{fault} ({fault_name}): {count:,} records")