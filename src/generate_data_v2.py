import gymnasium as gym
import numpy as np
import cv2
import os
from tqdm import tqdm

def generate_data(total_episodes=500, output_file="data/train_data_v2.npz"):
    print("🚀 Starting Data Generation (Version 2 - With Actions)...")
    
    # 创建环境
    try:
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
    except:
        env = gym.make("CarRacing-v2", render_mode="rgb_array")
    
    # 存储列表
    all_obs = []
    all_action = []
    
    for episode in tqdm(range(total_episodes), desc="Collecting"):
        obs, _ = env.reset()
        done = False
        
        # 临时缓存当前局的数据
        episode_obs = []
        episode_action = []
        
        # 同样跳过前 40 步
        for _ in range(40):
            env.step(env.action_space.sample())
            
        steps = 0
        while not done and steps < 100: # 每局跑 100 步
            action = env.action_space.sample() # 随机动作
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 1. 处理图片
            obs_resized = cv2.resize(obs, (64, 64))
            
            # 2. 存入临时列表
            episode_obs.append(obs_resized)
            episode_action.append(action)
            
            done = terminated or truncated
            steps += 1
        
        # 必须保证这一局的数据长度一致才保存
        if len(episode_obs) == len(episode_action):
            all_obs.extend(episode_obs)
            all_action.extend(episode_action)
            
    env.close()
    
    # 转换为 Numpy 数组
    all_obs = np.array(all_obs, dtype=np.uint8)
    all_action = np.array(all_action, dtype=np.float32)
    
    # 保存为压缩格式 .npz
    np.savez_compressed(output_file, obs=all_obs, action=all_action)
    print(f"✅ Data saved to {output_file}")
    print(f"   Obs Shape: {all_obs.shape}")       # 应为 (N, 64, 64, 3)
    print(f"   Action Shape: {all_action.shape}") # 应为 (N, 3)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_data()