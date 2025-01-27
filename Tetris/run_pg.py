import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from utils import preprocess_frame  # 預處理模組
from reward import calculate_reward
from model import PolicyNetwork, Trajectory

# 動作空間定義
SIMPLE_MOVEMENT = [
    ['NOOP'], ['left'], ['right'], ['down'], ['A'], ['B']
]
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 設定
VISUALIZE = True
LR = 0.0005
GAMMA = 0.99
TOTAL_TIMESTEPS = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立儲存權重的資料夾
CHECKPOINT_DIR = "ckpt_pg"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


# 初始化
obs_shape = (1, 84, 84)
n_actions = len(SIMPLE_MOVEMENT)
policy_net = PolicyNetwork(obs_shape, n_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)



trajectory = Trajectory()

# 訓練函數
def train(policy_net, optimizer, trajectory, gamma):
    # 計算折扣回報
    R = 0
    returns = []
    for r in trajectory.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 計算策略梯度損失
    states = torch.tensor(np.stack(trajectory.states), dtype=torch.float32, device=device)
    actions = torch.tensor(trajectory.actions, dtype=torch.int64, device=device)
    probs = policy_net(states)
    action_probs = probs.gather(1, actions.view(-1, 1)).squeeze()
    loss = -torch.sum(torch.log(action_probs) * returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    trajectory.clear()

# 主訓練迴圈
best_reward = float('-inf')  # 追蹤最佳獎勵

for episode in tqdm(range(TOTAL_TIMESTEPS), desc="Training Progress"):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    done = False
    cumulative_reward = 0
    prev_info = {}  # 初始化 prev_info
    combo_count = 0  # 初始化 combo_count

    while not done:
        
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        action_probs = policy_net(state_tensor).squeeze().cpu().detach().numpy()
        action = np.random.choice(n_actions, p=action_probs)

        next_state, reward, done, info = env.step(action)
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        # 更新 combo_count
        if info.get('rows_cleared', 0) > 0:
            combo_count += 1
        else:
            combo_count = 0

        cumulative_reward += calculate_reward(info, reward, prev_info, combo_count)
        # 儲存軌跡
        trajectory.store(state, action, reward)
        state = next_state
        prev_info = info.copy()  # 更新 prev_info

        if done:
            train(policy_net, optimizer, trajectory, GAMMA)
        
        if VISUALIZE:
            env.render()

    print(f"Episode {episode + 1} - Cumulative Reward: {cumulative_reward}")
    
    # 儲存權重
    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{episode+1}_reward_{int(cumulative_reward)}_custom_{int(cumulative_reward)}.pth")
        torch.save(policy_net.state_dict(), checkpoint_path)
        print(f"Saved new best model with reward {cumulative_reward} to {checkpoint_path}")

env.close()
