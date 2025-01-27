import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state 
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# # simple actions_dim = 7 
# SIMPLE_MOVEMENT = [
#     ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
# ]
#-----------------------------------------------------------------------------
#獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)
#例子:用來獎勵玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    #寫下蒐集到硬幣會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量

    total_reward += (info['coins'] - prev_info['coins']) * 10     #這裡是定義，如果玩家有蒐集到硬幣，則獎勵加10(這裡是可以自己去定義獎勵要給多少的)
    return total_reward

#用來鼓勵玩家進行跳躍或高度變化(因為有時前方有障礙物 會被卡住)
def distance_y_offset_reward(info, reward, prev_info):
    #寫下高度變化會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量
    
    y_diff = info['y_pos'] - prev_info['y_pos']                  #計算高度差異
    
    if y_diff > 0:                                               #如果有上升高度
        total_reward += 5                                       #給予正向獎勵10分
    elif y_diff < 0:                                            #如果有下降高度 
        total_reward += 2                                       #給予較小的正向獎勵3分
        
    return total_reward

#用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    #寫下前進會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量
    
    x_diff = info['x_pos'] - prev_info['x_pos']                  #計算水平位移差異
    
    if x_diff > 0:                                               #如果有向前移動
        total_reward += 10                            #給予正向獎勵,移動距離越大獎勵越多
    elif x_diff < 0:                                             #如果有後退
        total_reward -= 4                        #給予負向懲罰,後退距離越大懲罰越多
    else:                                                        #如果原地停留
        total_reward -= 10                                        #給予小幅負向懲罰
        
    return total_reward


#用來鼓勵玩家提高分數（例如擊敗敵人)
def monster_score_reward(info, reward, prev_info):
    
    #寫下分數增加會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量
    
    score_diff = info['score'] - prev_info['score']              #計算分數差異
    
    if score_diff > 0:                                           #如果分數有增加
        total_reward += score_diff * 1.5                         #給予正向獎勵,分數增加越多獎勵越多
    
    return total_reward

#用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info,reward):
    
    #寫下到達終點旗幟會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量
    
    if info['flag_get']:                                         #如果到達終點旗幟
        total_reward += 500                                      #給予大量正向獎勵500分
        
    return total_reward




#===============to do==========================================

def time_left_reward(info, reward, prev_info):

    total_reward = reward
    # 取得時間差（假設每一 frame 可能 time 減少 1 或更多）
    total_reward -= 0.1
    
    # 或者，也可以加一段：如果 time 所剩越多，代表行動有效率，可小幅獎勵
    # total_reward += info['time'] * 0.01  # 視需求調整

    return total_reward


def life_lost_penalty(info, reward, prev_info):
    total_reward = reward
    
    life_diff = info['life'] - prev_info['life']
    if life_diff < 0:
        # 失去一條命給予大幅懲罰
        total_reward -= abs(life_diff) * 200

    return total_reward

def update_combo_state(info, prev_info, combo_state, current_frame):


    score_diff = info['score'] - prev_info['score']
    # 假設分數增加大於某個門檻(例如 100 分)才代表真正擊殺敵人
    # (馬力歐遊戲分數事件很多，需要自己選合理門檻)
    if score_diff >= 100:
        # 如果距離上次擊殺事件未超過 time_window，累加 combo_count
        if current_frame - combo_state["last_event_frame"] <= combo_state["time_window"]:
            combo_state["combo_count"] += 1
        else:
            # 否則重置 combo_count 後再 +1
            combo_state["combo_count"] = 1

        # 記錄本次事件發生的 frame
        combo_state["last_event_frame"] = current_frame

    else:
        # 若這個 step 沒有擊殺敵人
        # 也可能要判斷距離 last_event_frame 是否超過 time_window 了
        if current_frame - combo_state["last_event_frame"] > combo_state["time_window"]:
            # 超過後就把 combo_count 歸零
            combo_state["combo_count"] = 0

    return combo_state

def combo_reward(info, reward, prev_info, combo_state):
    """
    根據 combo_state["combo_count"] 來給額外獎勵。
    也可細分：若 combo_count == 2, 3, 4... 分別給不同倍率的獎勵。
    """
    total_reward = reward

    # 若本 step 沒有增加 combo_count，就不用另外加獎勵。
    # 反之，如果 combo_count 有成長(通常在 update_combo_state 判斷)，可給 combo bonus。
    # 這裡示範：combo_count >= 2 才開始給 combo 獎勵。
    if combo_state["combo_count"] >= 2:
        # 例如：
        # combo 獎勵 = combo_count * 20
        # 或其他函數形式：2^(combo_count) - 1, 10*(combo_count^2), ...
        bonus = combo_state["combo_count"] * 20
        total_reward += bonus

    return total_reward

def stuck_penalty(info, reward, stuck_counter, stuck_threshold=30, penalty_value=50):
    """
    - stuck_counter: 代表已經連續多少 frame x_pos 沒有改變
    - stuck_threshold: 超過多少 frame 沒有明顯前進，就判定為卡住
    - penalty_value: 一旦判定卡住，給予一次性大懲罰
    """
    total_reward = reward
    # 若 stuck_counter 超過 threshold，直接懲罰
    if stuck_counter >= stuck_threshold:
        total_reward -= penalty_value
    return total_reward