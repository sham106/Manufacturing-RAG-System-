"""
Inventory Forecasting Module
Predicts paper consumption and generates purchase recommendations
"""

import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class InventoryForecaster:
    """
    Forecasting engine for paper inventory purchase predictions.
    Uses exponential smoothing for time-series forecasting.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the forecaster with configuration and data.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get settings
        self.inventory_data_path = self.config['data']['inventory_data']
        self.forecast_periods = self.config['forecasting']['forecast_periods']
        self.confidence_level = self.config['forecasting']['confidence_level']
        self.safety_factor = self.config['forecasting']['safety_factor']
        self.lead_time_days = self.config['forecasting']['lead_time_days']
        
        # Load data
        self.inventory_data = self._load_data()
        self.consumption_df = None
        self.models = {}  # Store trained models
        
        # Prepare data for forecasting
        self._prepare_data()
        
        print("✓ Inventory Forecaster initialized")
    
    def _load_data(self) -> Dict:
        """Load inventory and consumption data from JSON file."""
        print(f"Loading inventory data from {self.inventory_data_path}...")
        
        with open(self.inventory_data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data['consumption_history'])} consumption records")
        return data
    
    def _prepare_data(self):
        """
        Prepare consumption data for forecasting.
        Converts JSON to DataFrame and handles missing months.
        """
        print("Preparing consumption data for forecasting...")
        
        # Convert consumption history to DataFrame
        consumption = self.inventory_data['consumption_history']
        df = pd.DataFrame(consumption)
        
        # Create proper date column
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create unique identifier for each paper type/dimension combo
        df['paper_key'] = df['type'] + '_' + df['dimension_in_mm'].astype(str)
        
        self.consumption_df = df
        print(f"✓ Prepared {len(df)} consumption records")
        print(f"  Paper types: {df['paper_key'].nunique()} unique combinations")
    
    def train_models(self):
        """
        Train forecasting models for each paper type/dimension combination.
        Uses Exponential Smoothing which works well with limited data.
        """
        print("\nTraining forecasting models...")
        print("-" * 60)
        
        # Group by paper type and dimension
        paper_groups = self.consumption_df.groupby('paper_key')
        
        for paper_key, group in paper_groups:
            try:
                # Get consumption time series
                group = group.sort_values('date')
                consumption_series = group['consume_in_kg'].values
                
                # Need at least 3 data points to train
                if len(consumption_series) < 3:
                    print(f"⚠ {paper_key}: Insufficient data ({len(consumption_series)} points), using simple average")
                    # Store simple average as fallback
                    self.models[paper_key] = {
                        'type': 'simple_average',
                        'value': np.mean(consumption_series),
                        'std': np.std(consumption_series)
                    }
                    continue
                
                # Try exponential smoothing (good for trends)
                try:
                    model = ExponentialSmoothing(
                        consumption_series,
                        seasonal=None,  # Not enough data for seasonality
                        trend='add'     # Additive trend
                    )
                    fitted_model = model.fit()
                    
                    # Store the fitted model
                    self.models[paper_key] = {
                        'type': 'exponential_smoothing',
                        'model': fitted_model,
                        'last_value': consumption_series[-1],
                        'mean': np.mean(consumption_series),
                        'std': np.std(consumption_series)
                    }
                    
                    print(f"✓ {paper_key}: Exponential Smoothing trained ({len(consumption_series)} data points)")
                    
                except Exception as e:
                    # Fallback to simple average
                    print(f"⚠ {paper_key}: Using simple average (ES failed: {str(e)[:50]})")
                    self.models[paper_key] = {
                        'type': 'simple_average',
                        'value': np.mean(consumption_series),
                        'std': np.std(consumption_series)
                    }
                    
            except Exception as e:
                print(f"❌ {paper_key}: Failed to train model - {e}")
                continue
        
        print("-" * 60)
        print(f"✓ Trained models for {len(self.models)} paper types")
    
    def forecast(
        self, 
        paper_type: str, 
        dimension: int, 
        periods: int = None,
        confidence_level: float = None
    ) -> Dict:
        """
        Generate forecast for specific paper type and dimension.
        
        Args:
            paper_type: Type of paper (e.g., "Flute", "Test liner", "White top")
            dimension: Width in mm (e.g., 1700, 2000)
            periods: Number of months to forecast (default from config)
            confidence_level: Confidence level for intervals (default from config)
        
        Returns:
            Dictionary with forecast results
        """
        periods = periods or self.forecast_periods
        confidence_level = confidence_level or self.confidence_level
        
        paper_key = f"{paper_type}_{dimension}"
        
        # Check if model exists
        if paper_key not in self.models:
            return {
                "error": f"No model trained for {paper_type} {dimension}mm",
                "paper_type": paper_type,
                "dimension_mm": dimension,
                "available_types": list(self.models.keys())
            }
        
        model_info = self.models[paper_key]
        
        # Generate forecast based on model type
        if model_info['type'] == 'exponential_smoothing':
            # Use the trained model
            forecast_values = model_info['model'].forecast(periods)
            
            # Calculate confidence intervals (using historical std)
            std = model_info['std']
            z_score = 1.96 if confidence_level >= 0.95 else 1.645  # 95% or 90%
            margin = z_score * std
            
            lower_bound = forecast_values - margin
            upper_bound = forecast_values + margin
            
        else:  # simple_average
            # Use average with some uncertainty
            avg_value = model_info['value']
            std = model_info['std']
            
            forecast_values = np.array([avg_value] * periods)
            
            z_score = 1.96 if confidence_level >= 0.95 else 1.645
            margin = z_score * std
            
            lower_bound = forecast_values - margin
            upper_bound = forecast_values + margin
        
        # Ensure no negative values
        forecast_values = np.maximum(forecast_values, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        # Generate future dates (properly handle month boundaries)
        last_date = self.consumption_df[
            self.consumption_df['paper_key'] == paper_key
        ]['date'].max()
        
        forecast_dates = []
        for i in range(1, periods + 1):
            # Add months properly
            month = last_date.month + i
            year = last_date.year
            
            # Handle year rollover
            while month > 12:
                month -= 12
                year += 1
            
            # Create date for the first of the month
            forecast_date = datetime(year, month, 1)
            forecast_dates.append(forecast_date)
        
        # Format result
        result = {
            "paper_type": paper_type,
            "dimension_mm": dimension,
            "forecast_kg": [round(v, 2) for v in forecast_values],
            "lower_bound_kg": [round(v, 2) for v in lower_bound],
            "upper_bound_kg": [round(v, 2) for v in upper_bound],
            "forecast_dates": [d.strftime('%Y-%m') for d in forecast_dates],
            "confidence_level": confidence_level,
            "model_type": model_info['type'],
            "historical_average": round(model_info.get('mean', model_info.get('value', 0)), 2)
        }
        
        return result
    
    def get_purchase_recommendations(
        self,
        current_stock: Optional[Dict[str, int]] = None,
        lead_time_days: int = None,
        safety_factor: float = None
    ) -> List[Dict]:
        """
        Generate purchase recommendations based on forecasts.
        
        Args:
            current_stock: Dict of current stock {paper_key: stock_kg}
            lead_time_days: Supplier lead time (default from config)
            safety_factor: Safety stock multiplier (default from config)
        
        Returns:
            List of purchase recommendations
        """
        lead_time_days = lead_time_days or self.lead_time_days
        safety_factor = safety_factor or self.safety_factor
        
        # Calculate how many months of forecast we need based on lead time
        forecast_months = max(1, lead_time_days // 30)
        
        recommendations = []
        
        # Generate recommendation for each paper type
        for paper_key in self.models.keys():
            # Parse paper type and dimension
            parts = paper_key.rsplit('_', 1)
            paper_type = parts[0]
            dimension = int(parts[1])
            
            # Get forecast
            forecast_result = self.forecast(
                paper_type, 
                dimension, 
                periods=forecast_months
            )
            
            if 'error' in forecast_result:
                continue
            
            # Calculate required stock for lead time period
            forecast_consumption = sum(forecast_result['forecast_kg'])
            required_stock = forecast_consumption * safety_factor
            
            # Get current stock (default to 0 if not provided)
            current = current_stock.get(paper_key, 0) if current_stock else 0
            
            # Calculate order quantity
            order_quantity = max(0, required_stock - current)
            
            # Determine urgency
            if current <= 0:
                urgency = "CRITICAL"
            elif current < forecast_consumption * 0.5:
                urgency = "HIGH"
            elif current < required_stock:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"
            
            # Create recommendation
            rec = {
                "paper_type": paper_type,
                "dimension_mm": dimension,
                "current_stock_kg": round(current, 2),
                "forecast_consumption_kg": round(forecast_consumption, 2),
                "recommended_order_kg": round(order_quantity, 2),
                "urgency": urgency,
                "lead_time_days": lead_time_days,
                "safety_factor": safety_factor,
                "reasoning": self._generate_reasoning(
                    current, forecast_consumption, required_stock, urgency
                )
            }
            
            recommendations.append(rec)
        
        # Sort by urgency
        urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: urgency_order[x['urgency']])
        
        return recommendations
    
    def _generate_reasoning(
        self, 
        current: float, 
        forecast: float, 
        required: float, 
        urgency: str
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        
        if urgency == "CRITICAL":
            return f"Stock depleted! Forecasted consumption is {forecast:.0f}kg for lead time period. Order immediately."
        elif urgency == "HIGH":
            return f"Current stock ({current:.0f}kg) below 50% of forecasted consumption ({forecast:.0f}kg). Order soon."
        elif urgency == "MEDIUM":
            return f"Stock adequate but below safety level. Forecasted consumption: {forecast:.0f}kg."
        else:
            return f"Stock sufficient. Current: {current:.0f}kg, Forecast: {forecast:.0f}kg."
    
    def explain_forecast(self, paper_type: str, dimension: int) -> str:
        """
        Generate natural language explanation of forecast.
        
        Returns:
            Human-readable explanation string
        """
        paper_key = f"{paper_type}_{dimension}"
        
        if paper_key not in self.models:
            return f"No forecast available for {paper_type} {dimension}mm."
        
        # Get forecast
        forecast_result = self.forecast(paper_type, dimension, periods=3)
        
        if 'error' in forecast_result:
            return forecast_result['error']
        
        # Get historical data
        historical = self.consumption_df[
            self.consumption_df['paper_key'] == paper_key
        ].sort_values('date')
        
        model_info = self.models[paper_key]
        
        explanation = f"""
Forecast for {paper_type} {dimension}mm:

HISTORICAL PATTERN:
- Data points available: {len(historical)}
- Historical average consumption: {model_info.get('mean', model_info.get('value', 0)):.2f} kg/month
- Recent consumption (last record): {historical['consume_in_kg'].iloc[-1]:.2f} kg

FORECAST (Next 3 Months):
"""
        
        for i, (date, value, lower, upper) in enumerate(zip(
            forecast_result['forecast_dates'],
            forecast_result['forecast_kg'],
            forecast_result['lower_bound_kg'],
            forecast_result['upper_bound_kg']
        ), 1):
            explanation += f"- Month {i} ({date}): {value:.2f} kg (range: {lower:.2f} - {upper:.2f} kg)\n"
        
        explanation += f"""
MODEL DETAILS:
- Method: {model_info['type'].replace('_', ' ').title()}
- Confidence Level: {forecast_result['confidence_level']*100:.0f}%

INTERPRETATION:
"""
        
        avg_forecast = np.mean(forecast_result['forecast_kg'])
        historical_avg = model_info.get('mean', model_info.get('value', 0))
        
        if avg_forecast > historical_avg * 1.1:
            explanation += "- Forecast shows INCREASING trend compared to historical average\n"
        elif avg_forecast < historical_avg * 0.9:
            explanation += "- Forecast shows DECREASING trend compared to historical average\n"
        else:
            explanation += "- Forecast is STABLE, consistent with historical average\n"
        
        explanation += f"- Recommended safety stock: {avg_forecast * self.safety_factor:.2f} kg\n"
        
        return explanation
    
    def get_consumption_summary(self) -> Dict:
        """
        Generate summary statistics of historical consumption.
        
        Returns:
            Dictionary with summary metrics
        """
        summary = {
            "total_consumption_kg": round(self.consumption_df['consume_in_kg'].sum(), 2),
            "average_monthly_kg": round(self.consumption_df['consume_in_kg'].mean(), 2),
            "date_range": {
                "start": self.consumption_df['date'].min().strftime('%Y-%m'),
                "end": self.consumption_df['date'].max().strftime('%Y-%m')
            },
            "paper_types": {}
        }
        
        # Summary by paper type
        for paper_type in self.consumption_df['type'].unique():
            type_data = self.consumption_df[self.consumption_df['type'] == paper_type]
            summary["paper_types"][paper_type] = {
                "total_consumption_kg": round(type_data['consume_in_kg'].sum(), 2),
                "average_monthly_kg": round(type_data['consume_in_kg'].mean(), 2),
                "dimensions": sorted(type_data['dimension_in_mm'].unique().tolist())
            }
        
        return summary


# Test the forecaster
if __name__ == "__main__":
    print("="*60)
    print("Testing Inventory Forecasting Module")
    print("="*60)
    
    try:
        # Initialize forecaster
        forecaster = InventoryForecaster()
        
        # Train models
        print("\nTraining forecasting models...")
        forecaster.train_models()
        
        # Test forecast for Flute 1700mm
        print("\n" + "="*60)
        print("Test 1: Forecast for Flute 1700mm")
        print("="*60)
        forecast_result = forecaster.forecast("Flute", 1700, periods=3)
        print(json.dumps(forecast_result, indent=2))
        
        # Get explanation
        print("\n" + "="*60)
        print("Test 2: Forecast Explanation")
        print("="*60)
        explanation = forecaster.explain_forecast("Flute", 1700)
        print(explanation)
        
        # Get purchase recommendations
        print("\n" + "="*60)
        print("Test 3: Purchase Recommendations")
        print("="*60)
        
        # Simulate some current stock
        current_stock = {
            "Flute_1700": 5000,  # 5000 kg in stock
            "Test liner_2000": 3000
        }
        
        recommendations = forecaster.get_purchase_recommendations(
            current_stock=current_stock,
            lead_time_days=30
        )
        
        print(f"\nFound {len(recommendations)} recommendations:\n")
        for rec in recommendations[:5]:  # Show first 5
            print(f"{rec['urgency']} - {rec['paper_type']} {rec['dimension_mm']}mm:")
            print(f"  Current: {rec['current_stock_kg']}kg")
            print(f"  Forecast: {rec['forecast_consumption_kg']}kg")
            print(f"  Recommended Order: {rec['recommended_order_kg']}kg")
            print(f"  Reason: {rec['reasoning']}")
            print()
        
        # Get consumption summary
        print("="*60)
        print("Test 4: Consumption Summary")
        print("="*60)
        summary = forecaster.get_consumption_summary()
        print(json.dumps(summary, indent=2))
        
        print("\n" + "="*60)
        print("✓ Forecasting module working correctly!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()