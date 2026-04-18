import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tab_transformer import TabTransformer
from data_processing import preprocess_data

def visualize_explainability():
    print("Loading model and data for explainability...")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Cardinalities: Orientation (4), Distribution (6)
    model = TabTransformer(num_numerical=6, cat_cardinalities=[4, 6]).to(device)
    model.load_state_dict(torch.load('tab_transformer.pth'))
    model.eval()
    
    # Load Data (Test set)
    X_test = np.load('X_test.npy', allow_pickle=True)
    cat_idxs = [5, 7]
    num_idxs = [0, 1, 2, 3, 4, 6]
    
    def to_tensors(df_vals):
        x_cat = df_vals[:, cat_idxs].astype(np.int64)
        x_num = df_vals[:, num_idxs].astype(np.float32)
        return torch.tensor(x_num), torch.tensor(x_cat)
    
    # Take a sample of 10 items
    sample_indices = np.random.choice(len(X_test), 2, replace=False)
    X_sample = X_test[sample_indices]
    
    x_num, x_cat = to_tensors(pd.DataFrame(X_sample).values)
    x_num, x_cat = x_num.to(device), x_cat.to(device)
    
    # Forward pass
    with torch.no_grad():
        preds, attn_maps = model(x_num, x_cat)
        
    # attn_maps is a list of [batch, seq_len, seq_len] (if averaged? check MultiheadAttention)
    # Custom Layer returns: attn_weights from nn.MultiheadAttention
    # Shape: (batch, num_heads, target_len, source_len) OR (batch, target_len, source_len) depending on average_attn_weights
    # Default average_attn_weights=True returns (batch, L, S). 
    # But wait, I used need_weights=True.
    # In PyTorch 1.x, MultiheadAttention returns (attn_output, attn_output_weights).
    # IF average_attn_weights=True (default), weights are (batch, L, S).
    # Since I didn't specify average_attn_weights=False, it's averaged.
    
    # Categories are: Orientation, Glazing Distribution (2 items)
    # So seq_len should be 2.
    
    print("Extracting attention weights...")
    attention_weights = attn_maps[0].cpu().numpy() # Layer 1
    # Shape should be (batch, 2, 2)
    
    # Visualizing Attention for first sample
    feature_names = ['Orientation', 'Distribution']
    
    for i in range(len(sample_indices)):
        plt.figure(figsize=(6, 5))
        sns.heatmap(attention_weights[i], xticklabels=feature_names, yticklabels=feature_names, annot=True, cmap='viridis')
        plt.title(f'Attention Weights (Layer 1) - Sample {i+1}')
        plt.ylabel('Target Feature')
        plt.xlabel('Source Feature')
        plt.tight_layout()
        plt.savefig(f'attention_sample_{i+1}.png')
        
    print("Attention maps saved.")
    
    # Simple Feature Importance via Permutation (on numericals too)
    # This is "Explainable Thermal Influence Analysis"
    # We can perturb each feature and see effect on output
    
    print("Calculating Feature Importance via Permutation...")
    baseline_preds, _ = model(X_full_num, X_full_cat) # Use full dataset here as intended
    baseline_preds = baseline_preds.detach().cpu().numpy()
    
    importances = {}
    
    # Numerical features
    num_feature_names = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Glazing Area']
    
    # We need full dataset for robustness
    X_full_num, X_full_cat = to_tensors(pd.DataFrame(X_test).values)
    X_full_num, X_full_cat = X_full_num.to(device), X_full_cat.to(device)
    
    with torch.no_grad():
        base_out, _ = model(X_full_num, X_full_cat)
        base_mse = np.mean((base_out.cpu().numpy())**2) # Just checking magnitude change or variance
        
    # Perturb Numerical
    for i, name in enumerate(num_feature_names):
        temp_num = X_full_num.clone()
        temp_num[:, i] = temp_num[:, i][torch.randperm(len(temp_num))] # Shuffle
        with torch.no_grad():
            out, _ = model(temp_num, X_full_cat)
            diff = np.mean(np.abs(out.cpu().numpy() - base_out.cpu().numpy()))
            importances[name] = diff
            
    # Perturb Categorical
    for i, name in enumerate(feature_names):
        temp_cat = X_full_cat.clone()
        temp_cat[:, i] = temp_cat[:, i][torch.randperm(len(temp_cat))]
        with torch.no_grad():
            out, _ = model(X_full_num, temp_cat)
            diff = np.mean(np.abs(out.cpu().numpy() - base_out.cpu().numpy()))
            importances[name] = diff
            
    # Plot Importance
    plt.figure(figsize=(10, 6))
    names = list(importances.keys())
    values = list(importances.values())
    
    # Sort
    idx = np.argsort(values)[::-1]
    names = [names[i] for i in idx]
    values = [values[i] for i in idx]
    
    sns.barplot(x=values, y=names, palette='magma')
    plt.title('Feature Importance (Permutation Sensitivity)')
    plt.xlabel('Mean Prediction Shift')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("Feature importance saved.")

if __name__ == "__main__":
    visualize_explainability()
