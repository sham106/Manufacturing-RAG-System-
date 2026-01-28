"""
Manufacturing RAG System - Main Interface
Interactive chat interface for production data analysis and inventory forecasting
"""

import sys
from colorama import init, Fore, Style
from rag_pipeline import ManufacturingRAGPipeline
from forecasting_module import InventoryForecaster

# Initialize colorama for colored terminal output
init(autoreset=True)


class ManufacturingAssistant:
    """
    Main assistant that combines RAG and forecasting capabilities.
    """
    
    def __init__(self):
        """Initialize the assistant with RAG and forecasting modules."""
        print(Fore.CYAN + "\n" + "="*60)
        print(Fore.CYAN + "Manufacturing Intelligence Assistant")
        print(Fore.CYAN + "="*60)
        
        try:
            # Initialize RAG pipeline
            print(Fore.YELLOW + "\n[1/2] Initializing RAG pipeline...")
            self.rag = ManufacturingRAGPipeline()
            
            # Initialize forecasting module
            print(Fore.YELLOW + "\n[2/2] Initializing forecasting module...")
            self.forecaster = InventoryForecaster()
            self.forecaster.train_models()
            
            print(Fore.GREEN + "\n✓ System ready!")
            
        except Exception as e:
            print(Fore.RED + f"\n❌ Initialization failed: {e}")
            print(Fore.YELLOW + "\nTroubleshooting:")
            print("1. Make sure you ran: python vector_store.py build")
            print("2. Ensure LM Studio is running with a model loaded")
            print("3. Check that all data files exist")
            raise
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect if query is about forecasting or machine data.
        
        Returns:
            'forecast', 'purchase', or 'machine'
        """
        query_lower = query.lower()
        
        # Forecasting keywords
        forecast_keywords = [
            'forecast', 'predict', 'prediction', 'future', 'next month',
            'next quarter', 'consumption', 'will need', 'expected',
            'trend', 'projection'
        ]
        
        # Purchase recommendation keywords
        purchase_keywords = [
            'order', 'purchase', 'buy', 'stock', 'inventory',
            'restock', 'recommendation', 'should order', 'need to order'
        ]
        
        # Check for keywords
        if any(keyword in query_lower for keyword in purchase_keywords):
            return 'purchase'
        elif any(keyword in query_lower for keyword in forecast_keywords):
            return 'forecast'
        else:
            return 'machine'
    
    def handle_forecast_query(self, query: str) -> str:
        """Handle forecasting-related queries."""
        query_lower = query.lower()
        
        # Try to extract paper type and dimension from query
        paper_type = None
        dimension = None
        
        # Check for paper type (case insensitive)
        if 'test liner' in query_lower or 'testliner' in query_lower:
            paper_type = 'Test liner'
        elif 'white top' in query_lower or 'whitetop' in query_lower:
            paper_type = 'White top'
        elif 'flute' in query_lower:
            paper_type = 'Flute'
        
        # Extract dimension (look for 4-digit numbers between 1700-2200)
        import re
        dim_match = re.search(r'\b(1[7-9]\d{2}|2[0-2]\d{2})\b', query)
        if dim_match:
            dimension = int(dim_match.group(1))
        
        # If we have both paper type and dimension, get specific forecast
        if paper_type and dimension:
            try:
                # Get forecast
                forecast = self.forecaster.forecast(paper_type, dimension, periods=3)
                
                if 'error' in forecast:
                    return f"❌ {forecast['error']}"
                
                # Format response
                response = f"\n📊 Forecast for {paper_type} {dimension}mm:\n\n"
                
                for i, (date, value, lower, upper) in enumerate(zip(
                    forecast['forecast_dates'],
                    forecast['forecast_kg'],
                    forecast['lower_bound_kg'],
                    forecast['upper_bound_kg']
                ), 1):
                    response += f"Month {i} ({date}): {value:.0f} kg "
                    response += f"(range: {lower:.0f} - {upper:.0f} kg)\n"
                
                response += f"\nHistorical Average: {forecast['historical_average']:.0f} kg/month"
                response += f"\nModel: {forecast['model_type'].replace('_', ' ').title()}"
                response += f"\nConfidence: {forecast['confidence_level']*100:.0f}%"
                
                return response
                
            except Exception as e:
                return f"❌ Error generating forecast: {e}"
        
        # If paper type but no dimension, try to find a matching model
        elif paper_type:
            try:
                # Find any models for this paper type
                matching_models = [k for k in self.forecaster.models.keys() if paper_type in k]
                
                if not matching_models:
                    return f"❌ No consumption data available for {paper_type}.\nAvailable types: {list(self.forecaster.models.keys())}"
                
                # Use the first matching model
                paper_key = matching_models[0]
                parts = paper_key.rsplit('_', 1)
                dimension = int(parts[1])
                
                # Get forecast
                forecast = self.forecaster.forecast(paper_type, dimension, periods=3)
                
                if 'error' in forecast:
                    return f"❌ {forecast['error']}"
                
                # Format response
                response = f"\n📊 Forecast for {paper_type} {dimension}mm:\n\n"
                
                for i, (date, value, lower, upper) in enumerate(zip(
                    forecast['forecast_dates'],
                    forecast['forecast_kg'],
                    forecast['lower_bound_kg'],
                    forecast['upper_bound_kg']
                ), 1):
                    response += f"Month {i} ({date}): {value:,.0f} kg "
                    response += f"(range: {lower:,.0f} - {upper:,.0f} kg)\n"
                
                response += f"\nHistorical Average: {forecast['historical_average']:,.0f} kg/month"
                response += f"\nModel: {forecast['model_type'].replace('_', ' ').title()}"
                response += f"\nConfidence: {forecast['confidence_level']*100:.0f}%"
                
                return response
                
            except Exception as e:
                return f"❌ Error generating forecast: {e}"
        
        # If no specific paper type, show summary
        else:
            try:
                summary = self.forecaster.get_consumption_summary()
                
                response = "\n📊 Consumption Summary:\n\n"
                response += f"Total Historical Consumption: {summary['total_consumption_kg']:,.0f} kg\n"
                response += f"Average Monthly: {summary['average_monthly_kg']:,.0f} kg\n"
                response += f"Period: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
                
                response += "By Paper Type:\n"
                for paper_type, data in summary['paper_types'].items():
                    response += f"\n{paper_type}:"
                    response += f"\n  Total: {data['total_consumption_kg']:,.0f} kg"
                    response += f"\n  Average: {data['average_monthly_kg']:,.0f} kg/month"
                    response += f"\n  Dimensions: {', '.join(map(str, data['dimensions']))}mm"
                
                response += "\n\nTip: For specific forecasts, ask about a paper type and dimension."
                response += "\nExample: 'Forecast for Flute 1700mm'"
                
                return response
                
            except Exception as e:
                return f"❌ Error generating summary: {e}"
    
    def handle_purchase_query(self, query: str) -> str:
        """Handle purchase recommendation queries."""
        try:
            # For demo, assume some stock levels (in real system, would query database)
            current_stock = {
                "Flute_1700": 5000,
                "Test liner_2000": 3000,
                "White top_1950": 2000
            }
            
            recommendations = self.forecaster.get_purchase_recommendations(
                current_stock=current_stock,
                lead_time_days=30
            )
            
            # Filter for urgent ones first
            urgent = [r for r in recommendations if r['urgency'] in ['CRITICAL', 'HIGH']]
            
            response = "\n🛒 Purchase Recommendations:\n"
            response += "="*60 + "\n\n"
            
            if urgent:
                response += "⚠️  URGENT ORDERS NEEDED:\n\n"
                for rec in urgent:
                    response += f"[{rec['urgency']}] {rec['paper_type']} {rec['dimension_mm']}mm\n"
                    response += f"  Current Stock: {rec['current_stock_kg']:,.0f} kg\n"
                    response += f"  Forecasted Need: {rec['forecast_consumption_kg']:,.0f} kg\n"
                    response += f"  Recommended Order: {rec['recommended_order_kg']:,.0f} kg\n"
                    response += f"  Reason: {rec['reasoning']}\n\n"
            else:
                response += "✓ No urgent orders needed at this time.\n\n"
            
            # Show top 3 other recommendations
            others = [r for r in recommendations if r['urgency'] not in ['CRITICAL', 'HIGH']][:3]
            
            if others:
                response += "Other Recommendations:\n\n"
                for rec in others:
                    response += f"[{rec['urgency']}] {rec['paper_type']} {rec['dimension_mm']}mm: "
                    response += f"{rec['recommended_order_kg']:,.0f} kg\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error generating recommendations: {e}"
    
    def handle_machine_query(self, query: str) -> str:
        """Handle machine data queries using RAG."""
        try:
            result = self.rag.query(query, verbose=False)
            return result['answer']
        except Exception as e:
            return f"❌ Error processing query: {e}"
    
    def process_query(self, query: str) -> str:
        """
        Process user query and return response.
        Routes to appropriate handler based on query type.
        """
        # Detect query type
        query_type = self.detect_query_type(query)
        
        # Route to appropriate handler
        if query_type == 'forecast':
            return self.handle_forecast_query(query)
        elif query_type == 'purchase':
            return self.handle_purchase_query(query)
        else:
            return self.handle_machine_query(query)
    
    def run_interactive(self):
        """Run interactive chat session."""
        print("\n" + "="*60)
        print(Fore.CYAN + Style.BRIGHT + "Interactive Manufacturing Assistant")
        print("="*60)
        print(Fore.YELLOW + "\nCapabilities:")
        print("  • Machine performance analysis (OEE, faults, trends)")
        print("  • Inventory forecasting (consumption predictions)")
        print("  • Purchase recommendations (order planning)")
        print(Fore.YELLOW + "\nCommands:")
        print("  • 'help' - Show sample questions")
        print("  • 'quit' or 'exit' - End session")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(Fore.CYAN + "\n👋 Thank you for using Manufacturing Assistant!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Process query
                print(Fore.YELLOW + "\nAssistant: " + Style.RESET_ALL, end="")
                response = self.process_query(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print(Fore.CYAN + "\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(Fore.RED + f"\n❌ Error: {e}\n")
    
    def show_help(self):
        """Display sample questions."""
        print(Fore.CYAN + "\n📖 Sample Questions:\n")
        
        print(Fore.YELLOW + "Machine Performance:")
        print("  • What was the average OEE for MC001?")
        print("  • Which machine had the highest productivity?")
        print("  • Show me faults on MC002")
        print("  • What caused downtime yesterday?")
        
        print(Fore.YELLOW + "\nInventory Forecasting:")
        print("  • Forecast for Flute 1700mm")
        print("  • What's the predicted consumption for Test liner 2000mm?")
        print("  • Show consumption summary")
        
        print(Fore.YELLOW + "\nPurchase Planning:")
        print("  • What should we order?")
        print("  • Show purchase recommendations")
        print("  • Which paper types need restocking?")
        print()


def main():
    """Main entry point."""
    try:
        # Create assistant
        assistant = ManufacturingAssistant()
        
        # Run interactive session
        assistant.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(Fore.RED + f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()