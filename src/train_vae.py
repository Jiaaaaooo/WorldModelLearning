import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import sys
sys.path.append(".")

# 导入你自己写的模块
from src.vae import VAE
from src.dataset import CarRacingDataset

# --- ⚙️ 超参数设置 ---
BATCH_SIZE = 128      # 一次看 128 张图
LR = 1e-4             # 学习率
EPOCHS = 30           # 训练 30 轮
SAVE_DIR = "runs/vae_checkpoints" # 模型保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(recon_x, x, mu, logvar):
    """
    VAE 的 Loss = 重构误差 (MSE) + KL 散度 (KLD)
    """
    # 1. 重构误差: 生成的图和原图的“像素差异”
    # reduction='sum' 表示把这一批次所有像素的误差加起来
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL 散度: 限制潜变量 z 符合正态分布
    # 公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD, MSE, KLD

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 准备数据
    dataset = CarRacingDataset()
    # num_workers=4 利用 CPU 多核加速数据读取
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. 初始化模型
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"🚀 Start Training VAE on {DEVICE} | Batch Size: {BATCH_SIZE}")
    
    # 3. 训练循环
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, img in enumerate(dataloader):
            img = img.to(DEVICE) # 把数据搬到 3090 上
            
            # --- 标准三件套 ---
            optimizer.zero_grad()           # 1. 清空梯度
            recon_img, mu, logvar = model(img) # 2. 前向传播
            
            # 计算 Loss
            loss, mse, kld = loss_fn(recon_img, img, mu, logvar)
            
            loss.backward()                 # 3. 反向传播
            optimizer.step()                # 4. 更新参数
            
            total_loss += loss.item()
            
            # 每 20 个 Batch 打印一次 Log
            if batch_idx % 20 == 0:
                print(f"Epoch[{epoch+1}/{EPOCHS}] Batch[{batch_idx}] "
                      f"Loss: {loss.item()/len(img):.2f} "
                      f"(MSE: {mse.item()/len(img):.2f}, KLD: {kld.item()/len(img):.2f})")
        
        # 计算平均 Loss
        avg_loss = total_loss / len(dataset)
        print(f"====> Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # 每 5 轮保存一次模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/vae_epoch_{epoch+1}.pth")
            print(f"💾 Model saved to {SAVE_DIR}/vae_epoch_{epoch+1}.pth")

    print(f"✅ Training Finished in {(time.time()-start_time)/60:.1f} mins!")

if __name__ == "__main__":
    train()