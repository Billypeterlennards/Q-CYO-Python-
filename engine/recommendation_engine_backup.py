# engine/recommendation_engine.py
import sys
import os

# Add models to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML model
try:
    from models.yield_model import predict_yield
    USING_ML = True
    print("✅ Loaded trained ML model")
except ImportError as e:
    print(f"⚠️ Could not load ML model: {e}")
    USING_ML = False
    
    # Fallback function (your original formula)
    def predict_yield(r, t, s, c):
        bases = {'maize': 12.5, 'wheat': 8.2, 'rice': 10.8}
        soils = {'sandy': 0.85, 'clay': 1.15, 'loamy': 1.25}
        base = bases.get(c, 10.0)
        soil = soils.get(s, 1.0)
        return base * soil + r * 0.012 + max(0, t - 20) * 0.08

# Keep your other functions as they are (they're good!)
def assess_risk(r, t):
    score = 0
    if r < 50: score += 2
    elif r > 200: score += 3
    elif r > 150: score += 2
    if t < 10: score += 2
    elif t > 35: score += 3
    elif t > 30: score += 1
    if score >= 4: return 'HIGH'
    if score >= 2: return 'MEDIUM'
    return 'LOW'

def optimize_fertilizer(r, t, s, c):
    bases = {'maize': 280, 'wheat': 220, 'rice': 240}
    soils = {'sandy': 1.3, 'clay': 0.9, 'loamy': 1.0}
    base = bases.get(c, 250)
    soil = soils.get(s, 1.0)
    rf = 1 + (r - 100) * 0.001
    tf = 1 + (t - 25) * 0.005
    return base * soil * rf * tf

class RecommendationEngine:
    def __init__(self):
        self.using_ml = USING_ML
    
    def get_recommendation(self, rainfall, temp, soil, crop, area):
        try:
            # This will use ML model if available, otherwise fallback
            y = predict_yield(rainfall, temp, soil, crop)
            f = optimize_fertilizer(rainfall, temp, soil, crop)
            r = assess_risk(rainfall, temp)
            
            return {
                'yield_per_hectare': round(y, 2),
                'total_yield': round(y * area, 2),
                'fertilizer_kg_per_ha': round(f, 0),
                'weather_risk': r,
                'using_ml_model': self.using_ml,
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error',
                'using_ml_model': False
            }

# For backward compatibility with main.py
def generate(data):
    """Legacy function for main.py"""
    engine = RecommendationEngine()
    return engine.get_recommendation(
        data.get('rainfall', 120),
        data.get('temperature', 26),
        data.get('soil_type', 'sandy'),
        data.get('crop_type', 'maize'),
        data.get('area', 5.0)
    )