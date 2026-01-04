import pandas as pd
import joblib
import os

def load_international_data(filepath):
    """
    Load and preprocess agricultural data
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Tuple of (X_features, y_target)
    """
    print(f"📊 Loading data from: {filepath}")
    
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Find target column (common names for yield)
        target_candidates = ['Yield', 'yield', 'Yield_tons', 'production', 'target', 'Yield_kg_per_ha']
        target_col = None
        
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                print(f"   Found target column: {target_col}")
                break
        
        if target_col is None:
            # Assume last column is target
            target_col = df.columns[-1]
            print(f"   Using last column as target: {target_col}")
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Save feature names for prediction
        os.makedirs('saved_models', exist_ok=True)
        feature_path = 'saved_models/feature_names.pkl'
        joblib.dump(list(X.columns), feature_path)
        print(f"✅ Saved feature names to: {feature_path}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def get_feature_names():
    """Get feature names used during training"""
    try:
        return joblib.load('saved_models/feature_names.pkl')
    except:
        return None

def create_sample_data():
    """Create sample agricultural data for testing"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Rainfall_mm': np.random.uniform(50, 300, n_samples),
        'Temperature_C': np.random.uniform(15, 35, n_samples),
        'Soil_Type': np.random.choice(['sandy', 'loam', 'clay'], n_samples),
        'Crop_Type': np.random.choice(['maize', 'wheat', 'rice'], n_samples),
        'Soil_pH': np.random.uniform(5.5, 7.5, n_samples),
        'Organic_Matter_%': np.random.uniform(1.0, 4.0, n_samples),
        'Yield_tons_per_ha': np.random.uniform(5, 20, n_samples)
    }
    
    # Add some correlation
    df = pd.DataFrame(data)
    df['Yield_tons_per_ha'] = (
        df['Rainfall_mm'] * 0.02 +
        df['Temperature_C'] * 0.3 +
        (df['Soil_Type'] == 'loam') * 2.0 +
        (df['Crop_Type'] == 'maize') * 3.0 +
        np.random.normal(0, 1, n_samples)
    )
    
    # Ensure positive yields
    df['Yield_tons_per_ha'] = df['Yield_tons_per_ha'].clip(2, 25)
    
    return df

if __name__ == "__main__":
    print("Testing data preprocessing...")
    
    # Create and save sample data
    sample_df = create_sample_data()
    os.makedirs('data', exist_ok=True)
    sample_df.to_csv('data/yield_df.csv', index=False)
    
    print(f"✅ Created sample data: data/yield_df.csv")
    print(f"   Samples: {len(sample_df)}")
    print(f"   Columns: {list(sample_df.columns)}")
    print(f"\nSample data:")
    print(sample_df.head())
