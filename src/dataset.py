import torch
from torch.utils.data import Dataset
import numpy as np

class CarRacingDataset(Dataset):
    def __init__(self, data_path="data/train_data.npy"):
        # 1. 一次性把数据加载到内存
        # 10000张图片大约才 120MB，对于 3090 服务器的内存来说是九牛一毛
        print(f"Loading data from {data_path}...")
        self.data = np.load(data_path)
        print(f"Data loaded! Shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取第 idx 张图片: [64, 64, 3]
        img = self.data[idx]
        
        # 1. 归一化: 将像素值从 [0, 255] 变成 [0.0, 1.0] (float32)
        img = img.astype(np.float32) / 255.0
        
        # 2. 维度变换 (重点!):
        # Numpy/OpenCV 是 [Height, Width, Channel] -> (64, 64, 3)
        # PyTorch 需要    [Channel, Height, Width] -> (3, 64, 64)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        return img_tensor

if __name__ == "__main__":
    # 简单测试一下
    ds = CarRacingDataset()
    item = ds[0]
    print(f"Tensor Shape: {item.shape}, Max: {item.max()}, Min: {item.min()}")