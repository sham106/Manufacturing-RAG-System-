"""
Data Processor - Loads and prepares manufacturing data for RAG
This converts CSV data into text chunks that the AI can understand
"""

import pandas as pd
import yaml
import json
from typing import List, Dict
from datetime import datetime

class ManufacturingDataProcessor:
    """
    Processes manufacturing time-series data into chunks suitable for RAG.
    Converts rows of machine data into human-readable text descriptions.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data processor.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get file paths from config
        self.machine_data_path = self.config['data']['machine_data']
        self.inventory_data_path = self.config['data']['inventory_data']
        
        # Get fault code meanings from config
        self.fault_codes = self.config['manufacturing']['fault_codes']
        
        # Storage for processed data
        self.machine_df = None
        self.inventory_data = None
        self.text_chunks = []
    
    def load_machine_data(self) -> pd.DataFrame:
        """
        Load machine data from CSV file.
        
        Returns:
            DataFrame with machine production data
        """
        print(f"Loading machine data from {self.machine_data_path}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(self.machine_data_path)
            
            # Convert timestamp to datetime for easier processing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp')
            
            print(f"✓ Loaded {len(df)} records from {df['machine_id'].nunique()} machines")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            self.machine_df = df
            return df
            
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {self.machine_data_path}")
            print("Please run data_generator.py first to create the data file")
            raise
        except Exception as e:
            print(f"❌ ERROR loading machine data: {e}")
            raise
    
    def load_inventory_data(self) -> Dict:
        """
        Load inventory/consumption data from JSON file.
        
        Returns:
            Dictionary with inventory and consumption history
        """
        print(f"Loading inventory data from {self.inventory_data_path}...")
        
        try:
            with open(self.inventory_data_path, 'r') as f:
                data = json.load(f)
            
            print(f"✓ Loaded {len(data.get('consumption_history', []))} consumption records")
            
            self.inventory_data = data
            return data
            
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {self.inventory_data_path}")
            print("Please make sure paper_inventory_data.json is in the project folder")
            raise
        except Exception as e:
            print(f"❌ ERROR loading inventory data: {e}")
            raise
    
    def create_text_chunks(self, chunk_size_minutes: int = 30) -> List[str]:
        """
        Convert machine data into text chunks for RAG.
        
        This is the KEY function - it converts rows of numbers into 
        human-readable text that the AI can understand and search through.
        
        Args:
            chunk_size_minutes: Group data into chunks of this many minutes
            
        Returns:
            List of text strings, each describing a time period
        """
        if self.machine_df is None:
            raise ValueError("Machine data not loaded. Call load_machine_data() first.")
        
        print(f"Creating text chunks (grouping by {chunk_size_minutes} minutes)...")
        
        chunks = []
        
        # Group by machine
        for machine_id in self.machine_df['machine_id'].unique():
            machine_data = self.machine_df[self.machine_df['machine_id'] == machine_id].copy()
            
            # Create time-based groups
            machine_data['time_group'] = machine_data['timestamp'].dt.floor(f'{chunk_size_minutes}min')
            
            # Process each time group
            for time_group, group_data in machine_data.groupby('time_group'):
                chunk_text = self._create_chunk_text(machine_id, time_group, group_data)
                chunks.append(chunk_text)
        
        print(f"✓ Created {len(chunks)} text chunks from machine data")
        
        self.text_chunks = chunks
        return chunks
    
    def _create_chunk_text(self, machine_id: str, timestamp: datetime, data: pd.DataFrame) -> str:
        """
        Create a human-readable text description of a data chunk.
        
        This converts numbers into sentences like:
        "On MC001 at 2026-01-19 14:00, the machine was running at 85 boards/min..."
        
        Args:
            machine_id: Machine identifier
            timestamp: Time period start
            data: DataFrame rows for this chunk
            
        Returns:
            Text description of this time period
        """
        # Calculate statistics for this time period
        avg_speed = data['speed_board_per_min'].mean()
        avg_oee = data['oee_percent'].mean()
        avg_temp = data['temperature_c'].mean()
        avg_vibration = data['vibration_mm_s'].mean()
        
        # Count faults in this period
        faults = data[data['fault_code'] != '0']
        fault_count = len(faults)
        
        # Get most common feeder status
        feeder_status = data['feeder_status'].mode()[0] if len(data) > 0 else 'UNKNOWN'
        
        # Format timestamp nicely
        time_str = timestamp.strftime('%Y-%m-%d %H:%M')
        date_str = timestamp.strftime('%A, %B %d, %Y')  # e.g., "Monday, January 19, 2026"
        
        # Build the text description
        text_parts = []
        
        # Header with machine and time
        text_parts.append(f"=== {machine_id} Performance Report ===")
        text_parts.append(f"Date: {date_str}")
        text_parts.append(f"Time Period: {time_str}")
        text_parts.append("")
        
        # Operating status
        if avg_speed > 0:
            text_parts.append(f"Status: OPERATIONAL")
            text_parts.append(f"Production Speed: {avg_speed:.1f} boards per minute")
            text_parts.append(f"Overall Equipment Effectiveness (OEE): {avg_oee:.1f}%")
        else:
            text_parts.append(f"Status: STOPPED or FAULTED")
            text_parts.append(f"Production Speed: 0 boards per minute")
        
        # Feeder status
        text_parts.append(f"Feeder Status: {feeder_status}")
        
        # Operating conditions
        text_parts.append(f"")
        text_parts.append(f"Operating Conditions:")
        text_parts.append(f"- Temperature: {avg_temp:.1f}°C")
        text_parts.append(f"- Vibration: {avg_vibration:.2f} mm/s")
        
        # Fault information (important for RAG queries about problems)
        if fault_count > 0:
            text_parts.append(f"")
            text_parts.append(f"⚠ FAULTS DETECTED: {fault_count} fault(s) in this period")
            
            # List each unique fault
            unique_faults = faults['fault_code'].unique()
            for fault_code in unique_faults:
                fault_name = self.fault_codes.get(fault_code, "Unknown Fault")
                fault_occurrences = len(faults[faults['fault_code'] == fault_code])
                text_parts.append(f"- {fault_code}: {fault_name} ({fault_occurrences} occurrence(s))")
        else:
            text_parts.append(f"")
            text_parts.append(f"✓ No faults detected in this period")
        
        # Add production counter info
        if len(data) > 0:
            final_counter = data['counter_total'].iloc[-1]
            text_parts.append(f"")
            text_parts.append(f"Production Counter: {final_counter:,} total boards")
        
        # Join all parts with newlines
        return "\n".join(text_parts)
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the loaded data.
        Useful for understanding what data we have.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.machine_df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "total_records": len(self.machine_df),
            "machines": self.machine_df['machine_id'].unique().tolist(),
            "date_range": {
                "start": str(self.machine_df['timestamp'].min()),
                "end": str(self.machine_df['timestamp'].max())
            },
            "fault_statistics": {}
        }
        
        # Count each fault type
        fault_counts = self.machine_df['fault_code'].value_counts().to_dict()
        for fault_code, count in fault_counts.items():
            fault_name = self.fault_codes.get(fault_code, "Unknown")
            summary["fault_statistics"][fault_code] = {
                "name": fault_name,
                "count": int(count),
                "percentage": round(count / len(self.machine_df) * 100, 2)
            }
        
        # Average metrics per machine
        summary["machine_averages"] = {}
        for machine in self.machine_df['machine_id'].unique():
            machine_data = self.machine_df[self.machine_df['machine_id'] == machine]
            summary["machine_averages"][machine] = {
                "avg_speed": round(machine_data['speed_board_per_min'].mean(), 2),
                "avg_oee": round(machine_data['oee_percent'].mean(), 2),
                "avg_temperature": round(machine_data['temperature_c'].mean(), 2)
            }
        
        return summary


# Test the processor if run directly
if __name__ == "__main__":
    print("Testing Manufacturing Data Processor...")
    print("=" * 60)
    
    try:
        # Initialize processor
        processor = ManufacturingDataProcessor()
        
        # Load machine data
        print("\n1. Loading machine data...")
        df = processor.load_machine_data()
        print(f"   First few rows:")
        print(df.head())
        
        # Load inventory data
        print("\n2. Loading inventory data...")
        inventory = processor.load_inventory_data()
        print(f"   Found {len(inventory['paper_types'])} paper types")
        
        # Create text chunks
        print("\n3. Creating text chunks...")
        chunks = processor.create_text_chunks(chunk_size_minutes=30)
        
        # Show example chunk
        print("\n4. Example text chunk:")
        print("-" * 60)
        print(chunks[0])
        print("-" * 60)
        
        # Get summary
        print("\n5. Data summary:")
        summary = processor.get_data_summary()
        print(json.dumps(summary, indent=2))
        
        print("\n✓ Data processor is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()