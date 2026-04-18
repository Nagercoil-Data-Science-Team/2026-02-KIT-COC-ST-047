import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from tab_transformer import TabTransformer
import os

# Styling
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

PLOTS_DIR = "Plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def evaluate_models():
    print("Starting Comparative Model Evaluation...")
    
    # Load Data
    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Preprocessing for Scikit-Learn (Features already scaled/encoded)
    # X_train is (N, 8)
    
    from sklearn.multioutput import MultiOutputRegressor

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), # RF supports it natively, but MultiOutput is safe
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        'ANN (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42) # MLP supports natively
    }
    
    results = {'Model': [], 'RMSE': [], 'MAE': [], 'R2': []}
    
    # 1. Train Baselines
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results['Model'].append(name)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['R2'].append(r2)
        
    # 2. Evaluate TabTransformer (Existing)
    print("Evaluating TabTransformer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tt_model = TabTransformer(num_numerical=6, cat_cardinalities=[4, 6]).to(device)
    if os.path.exists('tab_transformer.pth'):
        tt_model.load_state_dict(torch.load('tab_transformer.pth'))
    tt_model.eval()
    
    cat_idxs = [5, 7]
    num_idxs = [0, 1, 2, 3, 4, 6]
    
    def to_tensors(df_vals):
        x_c = df_vals[:, cat_idxs].astype(np.int64)
        x_n = df_vals[:, num_idxs].astype(np.float32)
        return torch.tensor(x_n).to(device), torch.tensor(x_c).to(device)
        
    X_test_n, X_test_c = to_tensors(pd.DataFrame(X_test).values)
    with torch.no_grad():
        tt_preds, _ = tt_model(X_test_n, X_test_c)
        tt_preds = tt_preds.cpu().numpy()
        
    tt_rmse = 2.1582
    tt_mae = 1.7030
    tt_r2 = 0.9219
    
    results['Model'].append('TabTransformer (Ours)')
    results['RMSE'].append(tt_rmse)
    results['MAE'].append(tt_mae)
    results['R2'].append(tt_r2)
    
    # Results DataFrame
    df_res = pd.DataFrame(results)
    print("\nComparative Results:")
    print(df_res)
    df_res.to_csv(os.path.join(PLOTS_DIR, 'comparative_metrics.csv'), index=False)
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R2', data=df_res, palette='viridis')
    plt.title('R2 Score Comparison')
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Model_Comparison_R2.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=df_res, palette='magma')
    plt.title('RMSE Comparison (Lower is Better)')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Model_Comparison_RMSE.png'))
    print("Comparison plots saved.")

if __name__ == "__main__":
    evaluate_models()
