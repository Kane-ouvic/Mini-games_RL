import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

def rows_cleared_reward(info, reward):
    """
    獎勵玩家清除的行數。
    """
    total_reward = reward
    rows_cleared = info.get('rows_cleared', 0)  # 確保 info 包含 rows_cleared 信息

    # 獎勵按行數增加
    if rows_cleared > 0:
        # 單行清除：+10，雙行清除：+30，三行清除：+50，四行清除：+100（Tetris）
        total_reward += [0, 10, 30, 50, 100][rows_cleared]

    return total_reward


def empty_space_penalty(info, reward, prev_info):
    """
    懲罰未填充的空隙數量。
    """
    total_reward = reward
    empty_spaces = info.get('empty_spaces', 0)  # 假設 info 提供空隙數量

    # 每個空隙懲罰 -5
    total_reward -= empty_spaces * 5

    return total_reward

def pile_height_penalty(info, reward, prev_info):
    """
    懲罰堆積的最高高度。
    """
    total_reward = reward
    max_height = info.get('max_height', 0)  # 假設 info 提供堆積的最大高度

    # 堆積每增加一行，懲罰 -2
    total_reward -= max_height * 2

    return total_reward

def smoothness_reward(info, reward):
    """
    獎勵堆積表面的平滑度。
    """
    total_reward = reward
    surface_roughness = info.get('surface_roughness', 0)  # 假設 info 提供表面粗糙度

    # 粗糙度越小，獎勵越高
    total_reward += max(0, 20 - surface_roughness)

    return total_reward

def fast_drop_reward(info, reward, prev_info):
    """
    獎勵玩家快速放置方塊。
    """
    total_reward = reward
    fast_drop = info.get('fast_drop', False)  # 假設 info 提供是否使用快速下落的信息

    if fast_drop:
        total_reward += 10  # 使用快速下落給予額外獎勵

    return total_reward

def game_over_penalty(info, reward):
    """
    懲罰遊戲結束。
    """
    total_reward = reward
    game_over = info.get('game_over', False)  # 假設 info 提供遊戲結束信息

    if game_over:
        total_reward -= 500  # 大幅懲罰

    return total_reward

def combo_reward(info, reward, combo_count):
    """
    獎勵玩家連續清除的行數。
    """
    total_reward = reward
    if combo_count > 1:
        total_reward += combo_count * 20  # 每次 combo 增加額外獎勵

    return total_reward

def calculate_reward(info, reward, prev_info, combo_count):
    """
    計算所有獎勵和懲罰的總和。
    """
    total_reward = reward

    total_reward = rows_cleared_reward(info, total_reward)
    total_reward = empty_space_penalty(info, total_reward, prev_info)
    total_reward = pile_height_penalty(info, total_reward, prev_info)
    total_reward = smoothness_reward(info, total_reward)
    total_reward = fast_drop_reward(info, total_reward, prev_info)
    total_reward = game_over_penalty(info, total_reward)
    total_reward = combo_reward(info, total_reward, combo_count)

    return total_reward





