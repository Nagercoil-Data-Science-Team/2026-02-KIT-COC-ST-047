import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def generate_data(n_samples=768):
    """
    Generates synthetic data mimicking the UCI Energy Efficiency Dataset.
    Features:
    X1: Relative Compactness (0.62 - 0.98)
    X2: Surface Area (514.5 - 808.5)
    X3: Wall Area (245 - 416.5)
    X4: Roof Area (110.25 - 220.5)
    X5: Overall Height (3.5 - 7)
    X6: Orientation (2 - 5)
    X7: Glazing Area (0.0 - 0.4)
    X8: Glazing Area Distribution (0 - 5)
    
    Targets:
    Y1: Heating Load
    Y2: Cooling Load
    """
    np.random.seed(42)
    
    # Generate random features within approximate ranges
    X1 = np.random.uniform(0.62, 0.98, n_samples)
    X2 = -200 * X1 + 1000 + np.random.normal(0, 10, n_samples) # Inverse relation roughly
    X3 = np.random.uniform(245, 416.5, n_samples)
    X4 = np.random.choice([110.25, 120.5, 147.0, 220.5], n_samples)
    X5 = np.random.choice([3.5, 7.0], n_samples)
    X6 = np.random.randint(2, 6, n_samples) # 2,3,4,5
    X7 = np.random.choice([0.0, 0.1, 0.25, 0.4], n_samples)
    X8 = np.random.randint(0, 6, n_samples) # 0-5

    # Simulate Targets (Heating and Cooling Load)
    # Based on physics: Height, Glazing, and Compactness increase load
    Y1 = 10 + 50 * (X1 - 0.6) + 2 * (X5 - 3.5) + 20 * X7 + np.random.normal(0, 2, n_samples)
    Y2 = 12 + 55 * (X1 - 0.6) + 2.5 * (X5 - 3.5) + 25 * X7 + np.random.normal(0, 2, n_samples)

    data = pd.DataFrame({
        'Relative_Compactness': X1,
        'Surface_Area': X2,
        'Wall_Area': X3,
        'Roof_Area': X4,
        'Overall_Height': X5,
        'Orientation': X6,
        'Glazing_Area': X7,
        'Glazing_Area_Distribution': X8,
        'Heating_Load': Y1,
        'Cooling_Load': Y2
    })
    
    return data

def preprocess_data(df):
    # Separate targets
    targets = ['Heating_Load', 'Cooling_Load']
    y = df[targets].values
    X = df.drop(columns=targets)
    
    # Identify categorical and numerical columns
    # In this dataset, Orientation and Glazing Area Distribution are categorical
    cat_cols = ['Orientation', 'Glazing_Area_Distribution']
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # Scale numerical columns
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Encode categorical columns (Label Encoding for Embedding lookup)
    # Orientation: 2,3,4,5 -> 0,1,2,3
    # Distribution: 0-5 -> 0-5
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes.values
        
    return X, y, cat_cols, num_cols, scaler

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_data()
    print(f"Data generated. Shape: {df.shape}")
    
    print("Preprocessing data...")
    X, y, cat_cols, num_cols, scaler = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save processed arrays
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Data processed and saved.")
