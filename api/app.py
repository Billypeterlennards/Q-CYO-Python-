from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from engine.recommendation_engine import RecommendationEngine

app = Flask(__name__)
CORS(app)  # Enable CORS

# Create engine instance
recommendation_engine = RecommendationEngine()

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract parameters with defaults
        rainfall = data.get('rainfall', 120)
        temperature = data.get('temperature', 26)
        soil_type = data.get('soil_type', 'sandy')
        crop_type = data.get('crop_type', 'maize')
        area = data.get('area', 5)
        
        # Get recommendation
        result = recommendation_engine.get_recommendation(
            rainfall=rainfall,
            temperature=temperature,
            soil_type=soil_type,
            crop_type=crop_type,
            area=area
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': 'Invalid request',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'app': 'Quantum Crop Yield Optimizer (Q-CYO)',
        'version': '1.0',
        'endpoint': '/recommend (POST)',
        'parameters': {
            'rainfall': 'Rainfall in mm (number)',
            'temperature': 'Temperature in °C (number)',
            'soil_type': 'Soil type: sandy, clay, loamy, silty',
            'crop_type': 'Crop type: maize, wheat, rice, soybean, cotton, barley',
            'area': 'Farm area in hectares (number)'
        },
        'example': {
            'rainfall': 120,
            'temperature': 26,
            'soil_type': 'sandy',
            'crop_type': 'maize',
            'area': 5
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
