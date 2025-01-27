import os
import numpy as np
import torch
from tqdm import tqdm
import gym_tetris
from nes_py.wrappers import JoypadSpace
from utils import preprocess_frame
from reward import calculate_reward
from model import PolicyNetwork

# ========== 設定 ===========
MODEL_PATH = os.path.join("ckpt_pg", "step_2346_reward_267160_custom_267160.pth")  # 模型權重檔案路徑

# 動作空間定義
SIMPLE_MOVEMENT = [
    ['NOOP'], ['left'], ['right'], ['down'], ['A'], ['B']
]
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (1, 84, 84)
N_ACTIONS = len(SIMPLE_MOVEMENT)
VISUALIZE = True
TOTAL_EPISODES = 10


policy_net = PolicyNetwork(OBS_SHAPE, N_ACTIONS).to(device)

# ========== 載入模型權重 ===========
if os.path.exists(MODEL_PATH):
    try:
        model_weights = torch.load(MODEL_PATH, map_location=device)
        policy_net.load_state_dict(model_weights)
        policy_net.eval()
        print(f"成功載入模型: {MODEL_PATH}")
    except Exception as e:
        print(f"載入模型失敗: {e}")
        raise
else:
    raise FileNotFoundError(f"找不到模型檔案: {MODEL_PATH}")

# ========== 評估迴圈 ===========
for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    done = False
    total_reward = 0
    prev_info = {}
    combo_count = 0

    while not done:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            action_probs = policy_net(state_tensor).squeeze()
            action = torch.argmax(action_probs).item()

        next_state, reward, done, info = env.step(action)

        # 更新連續消除計數
        if info.get('rows_cleared', 0) > 0:
            combo_count += 1
        else:
            combo_count = 0

        # 計算獎勵
        reward = calculate_reward(info, reward, prev_info, combo_count)
        total_reward += reward

        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        
        state = next_state
        prev_info = info.copy()

        if VISUALIZE:
            env.render()

    print(f"Episode {episode}/{TOTAL_EPISODES} - Total Reward: {total_reward}")

env.close()
