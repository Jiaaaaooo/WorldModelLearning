import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=32):
        super(VAE, self).__init__()
        
        # [Encoder] 压缩图片: 3x64x64 -> 256x4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 4x4
            nn.ReLU(),
        )
        
        self.flatten_dim = 256 * 4 * 4
        
        # [Latent] 均值和方差
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # [Decoder] 还原图片
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # 必须用 Sigmoid 把像素限制在 0-1 之间
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧：让采样过程可导"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # 1. Encode
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        
        # 2. Latent Stats
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 3. Sample
        z = self.reparameterize(mu, logvar)
        
        # 4. Decode
        z_projected = self.fc_decode(z).view(-1, 256, 4, 4)
        recon_x = self.decoder(z_projected)
        
        return recon_x, mu, logvar

if __name__ == "__main__":
    # 简单的测试代码，确保跑通
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    vae = VAE().to(device)
    dummy_input = torch.randn(2, 3, 64, 64).to(device) # 模拟 batch=2 的图片
    
    recon, mu, logvar = vae(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {recon.shape}")
    
    assert recon.shape == dummy_input.shape
    print("✅ VAE Model check passed!")
    