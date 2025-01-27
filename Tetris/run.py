import os
import numpy as np
import random
import torch
import cv2
from tqdm import tqdm

import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame  # 預處理模組
from reward import *  # 獎勵函數模組
from model import CustomCNN  # 自定義模型
from DQN import DQN, ReplayMemory  # DQN模組

SIMPLE_MOVEMENT = [
    ['NOOP'],       # 不執行任何操作
    ['left'],       # 方塊左移
    ['right'],      # 方塊右移
    ['down'],       # 快速下降
    ['A'],          # 順時針旋轉
    ['B'],          # 逆時針旋轉
]

# ========== 動作空間定義 ==========
CUSTOM_MOVEMENT = SIMPLE_MOVEMENT  # 簡單動作空間，例如左移、右移、旋轉等

# ========== 設定 ==========
env = gym_tetris.make('TetrisA-v0')  # 建立 Tetris 環境
env = JoypadSpace(env, CUSTOM_MOVEMENT)

VISUALIZE = True
LR = 0.0001
BATCH_SIZE = 32
GAMMA = 0.99
MEMORY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
TARGET_UPDATE = 50
TOTAL_TIMESTEPS = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立儲存權重的資料夾
CKPT_DIR = "ckpt_test"
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# ========== DQN 初始化 ==========
obs_shape = (1, 84, 84)  # 狀態的形狀
n_actions = len(CUSTOM_MOVEMENT)  # 動作空間大小
model = CustomCNN  # 選擇模型
dqn = DQN(
    model=model,
    state_dim=obs_shape,
    action_dim=n_actions,
    learning_rate=LR,
    gamma=GAMMA,
    epsilon=EPSILON_START,
    target_update=TARGET_UPDATE,
    device=device
)
memory = ReplayMemory(MEMORY_SIZE)

# 追蹤最佳表現
best_reward = float('-inf')

# ========== 訓練開始 ==========
for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training Progress"):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    done = False
    cumulative_reward = 0
    prev_info = {}  # 初始化為空字典
    combo_count = 0

    while not done:
        # 選擇動作
        action = dqn.take_action(state)
        next_state, reward, done, info = env.step(action)
        
        # 狀態預處理
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        # 更新連續消除計數
        if info.get('rows_cleared', 0) > 0:
            combo_count += 1
        else:
            combo_count = 0

        # 更新累積獎勵
        cumulative_reward += calculate_reward(info, reward, prev_info, combo_count)

        # 存入記憶體
        memory.push(state, action, cumulative_reward, next_state, done)
        state = next_state
        prev_info = info.copy()  # 更新前一個狀態的info

        # 訓練
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            state_dict = {
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)

        # 更新 epsilon
        dqn.epsilon = max(EPSILON_END, EPSILON_START - timestep / TOTAL_TIMESTEPS)

        if VISUALIZE:
            env.render()

    print(f"Timestep {timestep} - Total Reward: {cumulative_reward}")

    # 儲存表現較好的權重
    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        # 儲存權重，檔名包含步數、獎勵和累積獎勵
        save_path = os.path.join(CKPT_DIR, f"step_{timestep}_reward_{int(reward)}_custom_{int(cumulative_reward)}.pth")
        torch.save(dqn.q_net.state_dict(), save_path)
        print(f"儲存新的最佳權重到: {save_path}")

env.close()
