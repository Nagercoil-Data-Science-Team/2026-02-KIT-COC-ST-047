import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from tab_transformer import TabTransformer
from scipy import stats

# --- STYLING SETUP ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

PLOTS_DIR = "Plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def get_data():
    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Merge for CV
    X = np.concatenate([X_train, X_test], axis=0) # (768, 8)
    y = np.concatenate([y_train, y_test], axis=0)
    
    return X, y

def to_tensors(df_vals, device):
    cat_idxs = [5, 7]
    num_idxs = [0, 1, 2, 3, 4, 6]
    x_cat = df_vals[:, cat_idxs].astype(np.int64)
    x_num = df_vals[:, num_idxs].astype(np.float32)
    return torch.tensor(x_num).to(device), torch.tensor(x_cat).to(device)

def train_and_evaluate(train_idx, val_idx, X, y, device, return_history=False):
    from torch.utils.data import DataLoader, TensorDataset

    # Data Split
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Tensors
    X_tr_n, X_tr_c = to_tensors(pd.DataFrame(X_train).values, device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    X_val_n, X_val_c = to_tensors(pd.DataFrame(X_val).values, device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # DataLoader
    train_ds = TensorDataset(X_tr_n, X_tr_c, y_tr)
    val_ds = TensorDataset(X_val_n, X_val_c, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # val_loader not strictly needed if val set is small enough, but good practice
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Model
    model = TabTransformer(num_numerical=6, cat_cardinalities=[4, 6]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_n, x_c, y_batch in train_loader:
            optimizer.zero_grad()
            out, _ = model(x_n, x_c)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_n.size(0)
            
        train_losses.append(running_loss / len(train_idx))
        
        # Val loss
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for x_n, x_c, y_batch in val_loader:
                val_out, _ = model(x_n, x_c)
                v_loss = criterion(val_out, y_batch)
                val_running_loss += v_loss.item() * x_n.size(0)
        val_losses.append(val_running_loss / len(val_idx))
    
    # Final Eval
    model.eval()
    all_preds = []
    with torch.no_grad():
         for x_n, x_c, y_batch in val_loader:
            preds, _ = model(x_n, x_c)
            all_preds.append(preds.cpu().numpy())
    preds = np.concatenate(all_preds)
        
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    
    if return_history:
        return rmse, mae, r2, train_losses, val_losses, y_val, preds
    return rmse, mae, r2

def main():
    print("Starting Detailed Evaluation...")
    X, y = get_data() # Full dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Cross Validation ---
    print("Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {'RMSE': [], 'MAE': [], 'R2': []}
    
    best_fold_data = None
    best_r2 = -float('inf')
    
    fold = 0
    for train_idx, val_idx in kf.split(X):
        rmse, mae, r2, t_loss, v_loss, y_true, y_pred = train_and_evaluate(train_idx, val_idx, X, y, device, return_history=True)
        cv_results['RMSE'].append(rmse)
        cv_results['MAE'].append(mae)
        cv_results['R2'].append(r2)
        
        if r2 > best_r2:
            best_r2 = r2
            best_fold_data = (t_loss, v_loss, y_true, y_pred)
            
        print(f"Fold {fold+1}: RMSE={rmse:.4f}, R2={r2:.4f}")
        fold += 1
        
    # Stats
    print("\nCross-Validation Statistics:")
    for metric, values in cv_results.items():
        mean_v = np.mean(values)
        std_v = np.std(values)
        print(f"{metric}: {mean_v:.4f} (+/- {std_v:.4f})")
        
    # --- PLOT 6: Cross-Validation Performance ---
    plt.figure(figsize=(8, 6))
    # data for boxplot
    # We plot RMSE and MAE. R2 is different scale.
    metrics_df = pd.DataFrame({
        'RMSE': cv_results['RMSE'],
        'MAE': cv_results['MAE']
    })
    sns.boxplot(data=metrics_df, palette='Set2')
    plt.title('Cross-Validation Performance')
    plt.ylabel('Error Value')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'CV_Performance.png'))
    plt.close()
    
    # --- Best Fold Analysis ---
    train_losses, val_losses, y_true, y_pred = best_fold_data
    
    # --- PLOT 1: Training vs Validation Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title('Training vs Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    # Grid removed as requested
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Train_Val_Loss.png'))
    plt.close()
    
    # --- PLOT 7: Learning Stability (Smoothed) ---
    plt.figure(figsize=(8, 6))
    series = pd.Series(train_losses)
    smoothed = series.rolling(window=5).mean()
    plt.plot(smoothed, label='Smoothed Train Loss', color='green', linewidth=2)
    plt.title('Learning Stability (Smoothed Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Learning_Stability.png'))
    plt.close()
    
    # --- Predictions ---
    # y_true shape (N, 2), y_pred shape (N, 2)
    y_heat_true = y_true[:, 0]
    y_heat_pred = y_pred[:, 0]
    y_cool_true = y_true[:, 1]
    y_cool_pred = y_pred[:, 1]
    
    # Save predicted values
    results_df = pd.DataFrame({
        'Actual_Heating': y_heat_true,
        'Predicted_Heating': y_heat_pred,
        'Actual_Cooling': y_cool_true,
        'Predicted_Cooling': y_cool_pred
    })
    results_df.to_csv(os.path.join(PLOTS_DIR, 'predicted_values.csv'), index=False)
    
    # --- PLOT 2: Actual vs Predicted Heating ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_heat_true, y_heat_pred, alpha=0.6, edgecolors='k')
    min_val = min(y_heat_true.min(), y_heat_pred.min())
    max_val = max(y_heat_true.max(), y_heat_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title('Actual vs Predicted Heating Load')
    plt.xlabel('Actual Heating Load')
    plt.ylabel('Predicted Heating Load')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Actual_vs_Pred_Heating.png'))
    plt.close()
    
    # --- PLOT 3: Actual vs Predicted Cooling ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_cool_true, y_cool_pred, alpha=0.6, edgecolors='k', color='orange')
    min_val = min(y_cool_true.min(), y_cool_pred.min())
    max_val = max(y_cool_true.max(), y_cool_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title('Actual vs Predicted Cooling Load')
    plt.xlabel('Actual Cooling Load')
    plt.ylabel('Predicted Cooling Load')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Actual_vs_Pred_Cooling.png'))
    plt.close()
    
    # --- Residuals ---
    res_heat = y_heat_true - y_heat_pred
    res_cool = y_cool_true - y_cool_pred
    all_res = np.concatenate([res_heat, res_cool])
    
    # --- PLOT 4: Prediction Error Distribution ---
    plt.figure(figsize=(8, 6))
    sns.histplot(all_res, kde=True, color='purple', bins=20)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Error_Distribution.png'))
    plt.close()
    
    # --- PLOT 5: Residuals vs Predicted ---
    plt.figure(figsize=(8, 6))
    # Combine predictions
    all_preds = np.concatenate([y_heat_pred, y_cool_pred])
    plt.scatter(all_preds, all_res, alpha=0.5, color='gray')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Load')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Residuals_vs_Predicted.png'))
    plt.close()
    
    # Save Metrics Text
    with open(os.path.join(PLOTS_DIR, 'final_metrics.txt'), 'w') as f:
        f.write("Cross-Validation Performance:\n")
        f.write(f"RMSE: {np.mean(cv_results['RMSE']):.4f} (+/- {np.std(cv_results['RMSE']):.4f})\n")
        f.write(f"MAE: {np.mean(cv_results['MAE']):.4f} (+/- {np.std(cv_results['MAE']):.4f})\n")
        f.write(f"R2: {np.mean(cv_results['R2']):.4f} (+/- {np.std(cv_results['R2']):.4f})\n")
        
    print("All plots and metrics generated in 'Plots' directory.")

if __name__ == "__main__":
    main()
