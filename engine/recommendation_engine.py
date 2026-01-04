import sys
import os

# Add parent directory to path to find other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import real models, fall back to mock implementations
try:
    from models.yield_model import predict_yield
    HAS_REAL_MODELS = True
except ImportError:
    HAS_REAL_MODELS = False
    
    # Mock implementations for development
    def predict_yield(rainfall, temperature, soil_type, crop_type):
        """Predict crop yield per hectare"""
        # Base yields for different crops (tons/hectare)
        base_yields = {
            'maize': 12.5, 'wheat': 8.2, 'rice': 10.8,
            'soybean': 9.5, 'cotton': 7.8, 'barley': 7.0
        }
        
        # Soil type multipliers
        soil_multipliers = {
            'sandy': 0.85, 'clay': 1.15, 'loamy': 1.25, 'silty': 1.10
        }
        
        base = base_yields.get(crop_type.lower(), 10.0)
        soil_factor = soil_multipliers.get(soil_type.lower(), 1.0)
        
        # Simple formula: base + rainfall effect + temperature effect
        rainfall_effect = rainfall * 0.012
        temp_effect = max(0, (temperature - 20) * 0.08)
        
        return base * soil_factor + rainfall_effect + temp_effect
    
    def assess_weather_risk(rainfall, temperature):
        """Assess weather risk level"""
        risk_score = 0
        
        # Rainfall risk
        if rainfall < 50:
            risk_score += 2  # Drought risk
        elif rainfall > 200:
            risk_score += 3  # Flood risk
        elif rainfall > 150:
            risk_score += 2
        
        # Temperature risk
        if temperature < 10:
            risk_score += 2  # Frost risk
        elif temperature > 35:
            risk_score += 3  # Heat stress
        elif temperature > 30:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def optimize_fertilizer(rainfall, temperature, soil_type, crop_type):
        """Optimize fertilizer amount (kg per hectare)"""
        # Base fertilizer requirements (kg/hectare)
        base_fertilizer = {
            'maize': 280, 'wheat': 220, 'rice': 240,
            'soybean': 180, 'cotton': 260, 'barley': 200
        }
        
        # Soil adjustments
        soil_adjustments = {
            'sandy': 1.3,  # Sandy soil needs more fertilizer
            'clay': 0.9,   # Clay soil retains nutrients better
            'loamy': 1.0,  # Loamy soil is ideal
            'silty': 1.1
        }
        
        base = base_fertilizer.get(crop_type.lower(), 250)
        soil_factor = soil_adjustments.get(soil_type.lower(), 1.0)
        
        # Adjust based on rainfall (more rain = more fertilizer needed)
        rainfall_factor = 1 + (rainfall - 100) * 0.001
        
        # Adjust based on temperature
        temp_factor = 1 + (temperature - 25) * 0.005
        
        return base * soil_factor * rainfall_factor * temp_factor


class RecommendationEngine:
    """Main engine for generating crop recommendations"""
    
    def __init__(self):
        self.has_real_models = HAS_REAL_MODELS
    
    def get_recommendation(self, rainfall, temperature, soil_type, crop_type, area):
        """
        Generate comprehensive crop recommendations
        
        Args:
            rainfall (float): Annual rainfall in mm
            temperature (float): Average temperature in °C
            soil_type (str): Type of soil (sandy, clay, loamy, silty)
            crop_type (str): Type of crop (maize, wheat, rice, etc.)
            area (float): Farm area in hectares
            
        Returns:
            dict: Recommendation results
        """
        try:
            # Calculate yield
            yield_per_ha = predict_yield(rainfall, temperature, soil_type, crop_type)
            total_yield = yield_per_ha * area
            
            # Get fertilizer recommendation
            fertilizer = optimize_fertilizer(rainfall, temperature, soil_type, crop_type)
            
            # Assess weather risk
            risk = assess_weather_risk(rainfall, temperature)
            
            return {
                'yield_per_hectare': round(yield_per_ha, 2),
                'total_yield': round(total_yield, 2),
                'fertilizer_kg_per_ha': round(fertilizer, 0),
                'weather_risk': risk,
                'using_real_models': self.has_real_models
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'using_real_models': self.has_real_models
            }


# Legacy function for backward compatibility
def generate(rainfall, temperature, soil_type='sandy', crop_type='maize', area=5):
    """
    Legacy function - creates engine instance and calls it
    
    This is for backward compatibility with old code
    """
    engine = RecommendationEngine()
    return engine.get_recommendation(rainfall, temperature, soil_type, crop_type, area)


# Example usage
if __name__ == '__main__':
    # Test the engine
    engine = RecommendationEngine()
    
    # Test case 1: Normal conditions
    result1 = engine.get_recommendation(120, 26, 'sandy', 'maize', 5)
    print('Test 1 - Normal conditions:')
    print(result1)
    
    # Test case 2: Extreme conditions
    result2 = engine.get_recommendation(250, 38, 'clay', 'rice', 3)
    print('\nTest 2 - Extreme conditions:')
    print(result2)
