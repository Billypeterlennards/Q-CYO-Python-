# models/yield_model.py
import joblib
import numpy as np
import pandas as pd
import os

class MLYieldPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model from saved_models folder"""
        try:
            # Get the absolute path to the model
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'saved_models', 'yield_model.pkl')
            
            if os.path.exists(model_path):
                print(f"🔍 Found model at: {model_path}")
                print(f"📏 File size: {os.path.getsize(model_path)} bytes")
                
                # Try to load the model
                self.model = joblib.load(model_path)
                print("✅ ML model loaded successfully!")
                self.is_loaded = True
                
                # Try to load scaler if exists
                scaler_path = os.path.join(current_dir, '..', 'saved_models', 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    print("✅ Scaler loaded")
                
                # Try to load encoder if exists
                encoder_path = os.path.join(current_dir, '..', 'saved_models', 'encoder.pkl')
                if os.path.exists(encoder_path):
                    self.encoder = joblib.load(encoder_path)
                    print("✅ Encoder loaded")
                    
            else:
                print(f"⚠️ Model not found at: {model_path}")
                print("Will use fallback formula instead")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def predict_with_ml(self, rainfall, temperature, soil_type, crop_type):
        """Make prediction using the trained ML model"""
        if not self.is_loaded or self.model is None:
            return None
        
        try:
            # Prepare the input data
            input_data = self._prepare_input(rainfall, temperature, soil_type, crop_type)
            
            # Make prediction
            prediction = self.model.predict(input_data)
            
            # Return the prediction
            return float(prediction[0])
            
        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            return None
    
    def _prepare_input(self, rainfall, temperature, soil_type, crop_type):
        """Prepare input features for the model"""
        # Create a DataFrame with the input data
        data = {
            'rainfall': [rainfall],
            'temperature': [temperature],
            'soil_type': [soil_type],
            'crop_type': [crop_type]
        }
        
        df = pd.DataFrame(data)
        
        # Apply preprocessing if available
        if self.scaler is not None:
            # Scale numerical features
            numerical_cols = ['rainfall', 'temperature']
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        if self.encoder is not None:
            # Encode categorical features
            categorical_cols = ['soil_type', 'crop_type']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = self.encoder.transform(df[col])
        
        return df
    
    def fallback_prediction(self, rainfall, temperature, soil_type, crop_type):
        """Fallback formula when ML model is not available"""
        bases = {'maize': 12.5, 'wheat': 8.2, 'rice': 10.8}
        soils = {'sandy': 0.85, 'clay': 1.15, 'loamy': 1.25}
        
        base = bases.get(crop_type.lower(), 10.0)
        soil = soils.get(soil_type.lower(), 1.0)
        
        return base * soil + rainfall * 0.012 + max(0, temperature - 20) * 0.08
    
    def predict(self, rainfall, temperature, soil_type, crop_type):
        """Public method to get prediction (tries ML first, then fallback)"""
        # Try ML model first
        ml_result = self.predict_with_ml(rainfall, temperature, soil_type, crop_type)
        
        if ml_result is not None:
            return ml_result
        else:
            # Use fallback formula
            return self.fallback_prediction(rainfall, temperature, soil_type, crop_type)

# Create a global instance
ml_predictor = MLYieldPredictor()

# Public function for easy import
def predict_yield(rainfall, temperature, soil_type, crop_type):
    """Use this function to predict yield (will use ML if available)"""
    return ml_predictor.predict(rainfall, temperature, soil_type, crop_type)

# Test the module
if __name__ == "__main__":
    print("🧪 Testing ML Model Loader...")
    test_result = predict_yield(120, 26, 'sandy', 'maize')
    print(f"📊 Prediction: {test_result} tons/hectare")
    print(f"🤖 Using ML Model: {ml_predictor.is_loaded}")