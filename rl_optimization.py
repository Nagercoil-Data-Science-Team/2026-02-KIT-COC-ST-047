import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from tab_transformer import TabTransformer
import pandas as pd

# --- Environment (Custom, no Gym dependency) ---
class BuildingRetrofitEnv:
    def __init__(self, model_path, scaler_path, num_numerical=6, cat_cardinalities=[4, 6]):
        
        # Load Surrogate Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TabTransformer(num_numerical, cat_cardinalities).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Action Space: Modify Glazing Area (4 levels) * Glazing Distribution (6 levels)
        self.num_actions = 4 
        self.num_features = 8
        
        # Initial State
        self.state = None
        self.original_loads = None
        
    def reset(self):
        # Randomly sample a building from X_test (simulating a new client)
        X_test = np.load('X_test.npy', allow_pickle=True)
        idx = np.random.randint(0, len(X_test))
        self.state = X_test[idx].copy() # 8 features
        
        # Calculate initial loads
        self.original_loads = self._get_loads(self.state)
        
        return self.state
        
    def _get_loads(self, state):
        cat_idxs = [5, 7]
        num_idxs = [0, 1, 2, 3, 4, 6]
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # (1, 8)
        
        x_c = state_tensor[:, cat_idxs].long().to(self.device)
        x_n = state_tensor[:, num_idxs].to(self.device)
        
        with torch.no_grad():
            preds, _ = self.model(x_n, x_c)
        return preds.cpu().numpy()[0] # [Heat, Cool]
        
    def step(self, action):
        # Map action to Glazing Area
        glazing_map = {0: 0.0, 1: 0.25, 2: 0.625, 3: 1.0} 
        
        current_glazing = self.state[6]
        new_glazing = glazing_map[action]
        
        self.state[6] = new_glazing
        
        # Calculate new loads
        new_loads = self._get_loads(self.state) # [Heat, Cool]
        total_energy = np.sum(new_loads)
        
        # Reward
        energy_saved = np.sum(self.original_loads) - total_energy
        cost = 1.0 if new_glazing != current_glazing else 0.0 
        
        reward = energy_saved - (cost * 2) 
        
        done = True # One step episode
        
        info = {
            'energy_saved': energy_saved,
            'new_load': total_energy,
            'original_load': np.sum(self.original_loads)
        }
        
        return self.state, reward, done, info

# --- DQN Agent ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train_rl():
    print("Starting RL Optimization (MODQN)...")
    
    env = BuildingRetrofitEnv(model_path='tab_transformer.pth', scaler_path='scaler.pkl')
    
    dqn = DQN(state_dim=8, action_dim=4)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    episodes = 500
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    rewards = []
    energy_savings = []
    
    for ep in range(episodes):
        state = env.reset()
        state_t = torch.FloatTensor(state)
        
        # Epsilon Greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                q_vals = dqn(state_t)
                action = torch.argmax(q_vals).item()
                
        next_state, reward, done, info = env.step(action)
        
        # Simple update (One step)
        target = reward 
        
        pred = dqn(state_t)[action]
        loss = criterion(pred, torch.tensor(target, dtype=torch.float32))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards.append(reward)
        energy_savings.append(info['energy_saved'])
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (ep+1) % 50 == 0:
            avg_rew = np.mean(rewards[-50:])
            avg_sav = np.mean(energy_savings[-50:])
            print(f"Episode {ep+1}, Avg Reward: {avg_rew:.2f}, Avg Energy Saved: {avg_sav:.2f}")
            
    # Plot RL Results
    plt.figure(figsize=(10, 5))
    plt.plot(pd.Series(rewards).rolling(50).mean())
    plt.title('RL Agent Learning Curve (Moving Avg Reward)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward (Energy Saving - Cost)')
    plt.savefig('rl_learning_curve.png')
    print("RL training complete. Plot saved.")

if __name__ == "__main__":
    train_rl()
