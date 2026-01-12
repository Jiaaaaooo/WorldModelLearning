import torch
import numpy as np
import os
from tqdm import tqdm
from vae import VAE # 导入你定义的模型

# 配置
VAE_CHECKPOINT = "runs/vae_checkpoints/vae_epoch_30.pth" # 确保路径对
DATA_PATH = "data/train_data_v2.npz"
OUTPUT_PATH = "data/processed_data.npz"
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Run generate_data.py first!")
        return

    # 1. 加载数据
    print(f"Loading raw data from {DATA_PATH}...")
    raw_data = np.load(DATA_PATH)
    images = raw_data['obs']    # (N, 64, 64, 3)
    actions = raw_data['action'] # (N, 3)
    N = len(images)
    
    # 2. 加载 VAE 模型
    print(f"Loading VAE model from {VAE_CHECKPOINT}...")
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT))
    vae.eval() # 评估模式，关闭 Dropout
    
    # 3. 开始转换 (Image -> Latent z)
    mu_list = []
    logvar_list = []
    
    print("🚀 Encoding images to latent space...")
    with torch.no_grad(): # 不计算梯度，节省显存
        # 批次处理，防止爆内存
        for i in tqdm(range(0, N, BATCH_SIZE)):
            # 取出一个 Batch
            batch_imgs = images[i : i + BATCH_SIZE]
            
            # 预处理: [0-255] -> [0.0-1.0] -> [B, 3, 64, 64]
            batch_tensor = torch.from_numpy(batch_imgs).float() / 255.0
            batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(DEVICE)
            
            # VAE Encoder 推理
            # 注意：我们要拿的是 mu 和 logvar，而不是重构的图片
            _, mu, logvar = vae(batch_tensor)
            
            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())
            
    # 4. 合并结果
    mu = np.concatenate(mu_list, axis=0)       # (N, 32)
    logvar = np.concatenate(logvar_list, axis=0) # (N, 32)
    
    print(f"✅ Processing Done!")
    print(f"   Latent Mu Shape: {mu.shape}")
    print(f"   Actions Shape: {actions.shape}")
    
    # 5. 保存处理后的数据
    np.savez_compressed(OUTPUT_PATH, mu=mu, logvar=logvar, action=actions)
    print(f"💾 Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process()