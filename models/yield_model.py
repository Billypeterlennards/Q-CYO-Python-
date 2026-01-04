# models/yield_model.py
"""
Quantum CYCYO - Yield Prediction Model
Combines trained ML model with formula-based fallback
"""

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------- TRAINING FUNCTION ----------
def train_yield_model(X, y):
    """
    Train ML model and save with feature names
    
    Args:
        X: Feature DataFrame
        y: Target values
    
    Returns:
        Trained RandomForest model
    """
    print(f"🎯 Training model with {X.shape[1]} features...")
    print(f"📋 Features: {list(X.columns)}")
    
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Fit the model
    model.fit(X, y)
    
    # Save feature names to model (for scikit-learn >= 0.24)
    model.feature_names_in_ = np.array(X.columns)
    
    print(f"✅ Model trained with {len(model.estimators_)} trees")
    print(f"📊 Feature importance:")
    for idx, (feature, importance) in enumerate(zip(X.columns, model.feature_importances_)):
        print(f"   {idx+1}. {feature}: {importance:.3f}")
    
    return model


# ---------- FORMULA-BASED PREDICTION (FALLBACK) ----------
def formula_predict_yield(rainfall, temperature, soil_type, crop_type):
    """
    Formula-based yield prediction (fallback when ML model is unavailable)
    
    Args:
        rainfall: mm
        temperature: °C
        soil_type: sandy/loam/clay
        crop_type: maize/wheat/rice
    
    Returns:
        Predicted yield in tons/hectare
    """
    # Base yields for different crops (tons/hectare)
    bases = {
        'maize': 12.5,
        'wheat': 8.2,
        'rice': 10.8,
        'barley': 7.5,
        'sorghum': 9.0,
        'soybean': 3.5
    }
    
    # Soil type multipliers
    soils = {
        'sandy': 0.85,   # Sandy soil has lower water/nutrient retention
        'loam': 1.25,    # Loam is ideal for most crops
        'clay': 1.15,    # Clay retains nutrients but may have drainage issues
        'silty': 1.10,   # Silty soil
        'peaty': 0.95,   # Peaty soil (acidic)
        'chalky': 0.90   # Chalky soil (alkaline)
    }
    
    # Get base yield for crop (default to 10 if unknown)
    base = bases.get(crop_type.lower(), 10.0)
    
    # Get soil multiplier (default to 1 if unknown)
    soil = soils.get(soil_type.lower(), 1.0)
    
    # Calculate yield using formula
    # Rainfall contribution: 0.012 tons per mm (optimal at 100-200mm)
    rainfall_contribution = rainfall * 0.012
    
    # Temperature contribution: optimal around 20-25°C
    # Below 20°C: penalty, above 20°C: bonus with diminishing returns
    temp_penalty = max(0, 20 - temperature) * 0.05  # Cold penalty
    temp_bonus = max(0, temperature - 20) * 0.08    # Warm bonus
    temp_contribution = temp_bonus - temp_penalty
    
    # Extreme temperature penalties
    if temperature > 35:
        temp_contribution -= 1.5  # Heat stress
    if temperature < 10:
        temp_contribution -= 2.0  # Cold stress
    
    # Extreme rainfall adjustments
    if rainfall < 50:
        rainfall_contribution *= 0.7  # Drought stress
    if rainfall > 250:
        rainfall_contribution *= 0.8  # Waterlogging
    
    # Calculate final yield
    yield_prediction = base * soil + rainfall_contribution + temp_contribution
    
    # Ensure non-negative yield
    return max(0.5, yield_prediction)


# ---------- ML PREDICTOR CLASS ----------
class YieldPredictor:
    """
    ML-based yield predictor using trained Random Forest model
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and feature names"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'saved_models', 'yield_model.pkl')
            
            if os.path.exists(model_path):
                print(f"🔍 Loading ML model from: {os.path.basename(model_path)}")
                self.model = joblib.load(model_path)
                self.is_loaded = True
                
                # Get feature names from model
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = list(self.model.feature_names_in_)
                    print(f"✅ Model loaded. Expected features: {self.feature_names}")
                else:
                    # Try to load from separate file
                    feature_path = os.path.join(current_dir, '..', 'saved_models', 'feature_names.pkl')
                    if os.path.exists(feature_path):
                        self.feature_names = joblib.load(feature_path)
                        print(f"✅ Features loaded from file: {self.feature_names}")
                    else:
                        print("⚠️ No feature names available. Using default mapping.")
                        self.feature_names = ['Rainfall', 'Temperature', 'Soil_Type', 'Crop_Type']
                
                # Test model with a sample prediction
                self._test_model()
                return True
                
            else:
                print(f"⚠️ Model not found: {model_path}")
                print("   Will use formula-based predictions")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _test_model(self):
        """Test the model with a sample input"""
        if not self.is_loaded:
            return
        
        try:
            # Create a test input
            test_features = {}
            for feature in self.feature_names:
                feature_lower = feature.lower()
                if 'rain' in feature_lower:
                    test_features[feature] = 120.0
                elif 'temp' in feature_lower:
                    test_features[feature] = 26.0
                elif 'soil' in feature_lower:
                    test_features[feature] = 'sandy'
                elif 'crop' in feature_lower:
                    test_features[feature] = 'maize'
                elif 'ph' in feature_lower:
                    test_features[feature] = 6.5
                elif 'organic' in feature_lower:
                    test_features[feature] = 2.0
                else:
                    test_features[feature] = 0.0
            
            # Make prediction
            test_df = pd.DataFrame([test_features])[self.feature_names]
            test_prediction = self.model.predict(test_df)
            print(f"🧪 Model test prediction: {test_prediction[0]:.2f} tons/ha")
            
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
    
    def _map_input_to_features(self, rainfall, temperature, soil_type, crop_type):
        """
        Map input parameters to model's expected features
        
        Args:
            rainfall: mm
            temperature: °C
            soil_type: sandy/loam/clay
            crop_type: maize/wheat/rice
        
        Returns:
            Dictionary mapping feature names to values
        """
        if not self.feature_names:
            # Default mapping if no feature names
            return {
                'Rainfall': rainfall,
                'Temperature': temperature,
                'Soil_Type': soil_type,
                'Crop_Type': crop_type
            }
        
        # Create mapping based on actual feature names
        feature_dict = {}
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            # Map based on feature name patterns
            if any(keyword in feature_lower for keyword in ['rain', 'precip', 'water']):
                feature_dict[feature] = rainfall
            elif any(keyword in feature_lower for keyword in ['temp', 'temperature', 'heat']):
                feature_dict[feature] = temperature
            elif any(keyword in feature_lower for keyword in ['soil', 'dirt', 'earth']):
                feature_dict[feature] = soil_type
            elif any(keyword in feature_lower for keyword in ['crop', 'plant', 'variety']):
                feature_dict[feature] = crop_type
            else:
                # Set sensible defaults for other features
                if 'ph' in feature_lower:
                    feature_dict[feature] = 6.5  # Neutral pH
                elif any(keyword in feature_lower for keyword in ['organic', 'matter', 'carbon']):
                    feature_dict[feature] = 2.0  # 2% organic matter
                elif any(keyword in feature_lower for keyword in ['nitrogen', 'n_']):
                    feature_dict[feature] = 50.0  # Medium nitrogen level
                elif any(keyword in feature_lower for keyword in ['phosphorus', 'p_', 'phosphate']):
                    feature_dict[feature] = 30.0  # Medium phosphorus
                elif any(keyword in feature_lower for keyword in ['potassium', 'k_', 'potash']):
                    feature_dict[feature] = 40.0  # Medium potassium
                elif 'altitude' in feature_lower or 'elevation' in feature_lower:
                    feature_dict[feature] = 100.0  # 100m altitude
                else:
                    feature_dict[feature] = 0.0  # Default for unknown features
        
        return feature_dict
    
    def predict(self, rainfall, temperature, soil_type, crop_type):
        """
        Make prediction using the trained ML model
        
        Args:
            rainfall: mm
            temperature: °C
            soil_type: sandy/loam/clay
            crop_type: maize/wheat/rice
        
        Returns:
            Predicted yield in tons/hectare, or None if prediction fails
        """
        if not self.is_loaded:
            print("⚠️ ML model not loaded, cannot make prediction")
            return None
        
        try:
            # Map inputs to expected features
            feature_dict = self._map_input_to_features(rainfall, temperature, soil_type, crop_type)
            
            # Ensure all expected features are present
            missing_features = []
            for feature in self.feature_names:
                if feature not in feature_dict:
                    missing_features.append(feature)
                    feature_dict[feature] = 0.0  # Fill with default
            
            if missing_features:
                print(f"⚠️ Missing features filled with defaults: {missing_features}")
            
            # Create DataFrame in correct feature order
            input_data = pd.DataFrame([feature_dict])[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(input_data)
            predicted_yield = float(prediction[0])
            
            # Ensure reasonable bounds
            if predicted_yield < 0:
                print(f"⚠️ Model predicted negative yield ({predicted_yield}), clipping to 0.5")
                predicted_yield = 0.5
            elif predicted_yield > 50:  # Unrealistically high
                print(f"⚠️ Model predicted unrealistic yield ({predicted_yield}), capping at 30")
                predicted_yield = min(predicted_yield, 30.0)
            
            print(f"🤖 ML prediction: {predicted_yield:.2f} tons/ha")
            return predicted_yield
            
        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            if 'feature_dict' in locals():
                print(f"   Input mapping: {feature_dict}")
            return None


# ---------- HYBRID PREDICTOR (ML + FORMULA) ----------
class HybridYieldPredictor:
    """
    Hybrid predictor that tries ML first, then falls back to formula
    """
    
    def __init__(self):
        self.ml_predictor = YieldPredictor()
        self.use_ml = self.ml_predictor.is_loaded
    
    def predict(self, rainfall, temperature, soil_type, crop_type):
        """
        Smart prediction: Try ML first, fallback to formula
        
        Args:
            rainfall: mm
            temperature: °C
            soil_type: sandy/loam/clay
            crop_type: maize/wheat/rice
        
        Returns:
            Tuple of (predicted_yield, method_used)
        """
        # Try ML prediction first
        if self.use_ml:
            ml_result = self.ml_predictor.predict(rainfall, temperature, soil_type, crop_type)
            if ml_result is not None:
                return ml_result, 'ml_model'
        
        # Fallback to formula
        formula_result = formula_predict_yield(rainfall, temperature, soil_type, crop_type)
        return formula_result, 'formula'


# ---------- GLOBAL INSTANCE AND PUBLIC FUNCTION ----------
# Create global hybrid predictor instance
yield_predictor = HybridYieldPredictor()

def predict_yield(rainfall, temperature, soil_type, crop_type):
    """
    Public function to predict yield (uses hybrid approach)
    
    Args:
        rainfall: mm
        temperature: °C
        soil_type: sandy/loam/clay
        crop_type: maize/wheat/rice
    
    Returns:
        Tuple of (predicted_yield, method_used)
    """
    return yield_predictor.predict(rainfall, temperature, soil_type, crop_type)


# ---------- MODEL EVALUATION FUNCTIONS ----------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'samples': len(y_test)
    }
    
    return metrics


# ---------- MAIN EXECUTION (FOR TESTING) ----------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING YIELD MODEL")
    print("="*60)
    
    # Test predictions with different methods
    test_cases = [
        (120, 26, 'sandy', 'maize'),
        (150, 28, 'loam', 'wheat'),
        (200, 30, 'clay', 'rice'),
        (80, 22, 'sandy', 'barley'),
        (250, 32, 'loam', 'sorghum')
    ]
    
    print("\n📊 Predictions:")
    for i, (rainfall, temp, soil, crop) in enumerate(test_cases, 1):
        yield_pred, method = predict_yield(rainfall, temp, soil, crop)
        print(f"{i}. {crop} on {soil} soil:")
        print(f"   Rainfall: {rainfall}mm, Temp: {temp}°C")
        print(f"   Prediction: {yield_pred:.2f} tons/ha ({method})")
        print()
    
    # Check model status
    print(f"🔧 Model Status:")
    print(f"   ML Model Loaded: {yield_predictor.use_ml}")
    if yield_predictor.use_ml:
        print(f"   Features Expected: {yield_predictor.ml_predictor.feature_names}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)