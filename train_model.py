# Create clean train_model.py in root (since your Profile folder might have issues)

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def load_data(filepath):
    """Load and prepare data"""
    print(f"📊 Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("   Creating sample data...")
        create_sample_data(filepath)
    
    df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Find target column
    target_candidates = ['Yield', 'yield', 'Yield_tons', 'production', 'target']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    print(f"   Target column: {target_col}")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Convert categorical to numeric
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y, target_col

def create_sample_data(filepath):
    """Create sample data if none exists"""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'Rainfall': np.random.uniform(50, 300, n_samples),
        'Temperature': np.random.uniform(15, 35, n_samples),
        'Soil_Type': np.random.choice(['sandy', 'loam', 'clay'], n_samples),
        'Crop_Type': np.random.choice(['maize', 'wheat', 'rice'], n_samples),
        'Soil_pH': np.random.uniform(5.5, 7.5, n_samples),
        'Yield': np.random.uniform(5, 25, n_samples)
    }
    
    # Add realistic relationships
    df = pd.DataFrame(data)
    df['Yield'] = (
        (df['Rainfall'] * 0.015) +
        (df['Temperature'] * 0.25) +
        (df['Soil_Type'] == 'loam') * 2.5 +
        (df['Soil_Type'] == 'clay') * 1.5 +
        (df['Crop_Type'] == 'maize') * 3.0 +
        (df['Crop_Type'] == 'rice') * 2.0 +
        np.random.normal(0, 1.5, n_samples)
    )
    
    df['Yield'] = df['Yield'].clip(3, 30)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Created sample data: {filepath}")
    
    return df

def train_model(X, y):
    """Train Random Forest model"""
    print(f"🎯 Training model with {X.shape[1]} features...")
    print(f"   Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Performance:")
    print(f"   Mean Absolute Error: {mae:.2f} tons/ha")
    print(f"   R² Score: {r2:.3f}")
    print(f"   Test samples: {len(X_test)}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔝 Top 5 important features:")
        importances = model.feature_importances_
        features = X.columns
        
        for idx in np.argsort(importances)[-5:][::-1]:
            print(f"   {features[idx]}: {importances[idx]:.3f}")
    
    return model

def save_model_and_features(model, X, output_dir='saved_models'):
    """Save model and feature names"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'yield_model.pkl')
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_path = os.path.join(output_dir, 'feature_names.pkl')
    joblib.dump(list(X.columns), feature_path)
    
    # Save model info
    info = {
        'model_type': 'RandomForestRegressor',
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'n_estimators': model.n_estimators,
        'model_size_kb': os.path.getsize(model_path) / 1024
    }
    
    info_path = os.path.join(output_dir, 'model_info.pkl')
    joblib.dump(info, info_path)
    
    print(f"\n💾 Saved files:")
    print(f"   Model: {model_path}")
    print(f"   Features: {feature_path}")
    print(f"   Info: {info_path}")
    print(f"   Model size: {info['model_size_kb']:.1f} KB")
    
    return model_path, feature_path

def test_model_prediction(model, X_sample):
    """Test model with sample input"""
    print(f"\n🧪 Testing model prediction...")
    
    # Create a sample input
    sample_input = X_sample.iloc[0:1].copy()
    
    try:
        prediction = model.predict(sample_input)
        print(f"   Sample prediction: {prediction[0]:.2f} tons/ha")
        
        # Show input features
        print(f"   Input features:")
        for col, val in sample_input.iloc[0].items():
            print(f"     {col}: {val}")
            
        return prediction[0]
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("🌾 QUANTUM CYCYO - MODEL TRAINING")
    print("="*60)
    
    # File paths
    data_file = 'data/yield_df.csv'
    
    # Load data
    try:
        X, y, target_col = load_data(data_file)
        print(f"✅ Data loaded successfully")
        print(f"   Target: {target_col}")
        print(f"   Range: {y.min():.1f} to {y.max():.1f} tons/ha")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Train model
    print("\n" + "-"*40)
    model = train_model(X, y)
    
    # Save model
    print("\n" + "-"*40)
    model_path, feature_path = save_model_and_features(model, X)
    
    # Test prediction
    print("\n" + "-"*40)
    test_model_prediction(model, X)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Start the API: python api/app.py")
    print("2. Test prediction: python -c \"from models.yield_model import predict_yield; print(predict_yield(120, 26, 'sandy', 'maize'))\"")
    print("3. Verify ML is working in API responses")
    print("="*60)

if __name__ == "__main__":
    main()
