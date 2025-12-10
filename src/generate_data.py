import gymnasium as gym
import numpy as np
import cv2
import os
from tqdm import tqdm

def generate_data(total_episodes=200, output_file="data/train_data.npy"):
    print("🚀 Starting Data Generation...")
    
    # 1. 创建调试目录 (用来存图片给你看)
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    # 2. 创建环境 (注意这里改成了 v3)
    try:
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
    except:
        print("⚠️ CarRacing-v3 failed, falling back to v2")
        env = gym.make("CarRacing-v2", render_mode="rgb_array")
    
    data = []
    total_frames_saved = 0
    
    for episode in tqdm(range(total_episodes), desc="Collecting Episodes"):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # 每一局游戏其实前 50 步都是没用的（镜头在拉近），我们跳过前 40 步
        # 这是一个小 Trick，让数据质量更高
        for _ in range(40):
            env.step(env.action_space.sample())
        
        while not done and steps < 50:
            action = env.action_space.sample() # 随机瞎跑
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- 核心处理 ---
            # 1. Resize 到 64x64 (适配我们的 VAE)
            obs_resized = cv2.resize(obs, (64, 64))
            
            # 2. [可视化] 保存前 20 张图给你检查！
            if total_frames_saved < 20:
                # 注意：Gym 输出是 RGB，OpenCV 保存需要 BGR，所以要转换颜色
                debug_img = cv2.cvtColor(obs_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{debug_dir}/frame_{total_frames_saved}.png", debug_img)
                total_frames_saved += 1
            
            # 3. 存入列表
            data.append(obs_resized)
            
            done = terminated or truncated
            steps += 1
            
    env.close()
    
    # 保存数据
    data_np = np.array(data, dtype=np.uint8)
    np.save(output_file, data_np)
    print(f"✅ Data saved to {output_file}. Shape: {data_np.shape}")
    print(f"🖼️  Check the '{debug_dir}' folder to see real images!")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_data()