import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, action_dim=3, hidden_size=256, num_gaussians=5):
        super(MDNRNN, self).__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        
        # 1. LSTM Core
        # 输入: z (32) + action (3) = 35
        self.lstm = nn.LSTM(input_size=z_dim + action_dim, 
                            hidden_size=hidden_size, 
                            batch_first=True)
        
        # 2. MDN Heads (输出层)
        # 我们不直接预测 z，而是预测 z 的分布参数 (Mixture of Gaussians)
        # 对于 z 的每一个维度，我们都需要预测:
        # - pi: 选哪个高斯核的概率 (num_gaussians)
        # - mu: 高斯核的均值 (num_gaussians)
        # - sigma: 高斯核的标准差 (num_gaussians)
        
        # 实际上为了简化，World Model 论文通常把 z 当做一个整体向量处理
        # 或者对每个 z 维度独立建模。这里我们采用标准论文实现方式：
        # 输出总维度 = num_gaussians * (z_dim * 2 + 1) 不太对
        # 简化版：我们只预测每个 z 维度的 5 个高斯分布
        
        self.fc_out = nn.Linear(hidden_size, num_gaussians * (2 * z_dim + 1)) 
        # 解释上面的维度公式:
        # pi: 1 个值 (权重) * num_gaussians
        # mu: z_dim 个值 * num_gaussians
        # sigma: z_dim 个值 * num_gaussians
        # 但通常为了简单，我们会把 pi 设为标量。
        # 这里我们使用 Ha 等人的简化版：输出 num_gaussians 组 (mu, logsigma, pi)
        
        # 修正：为了方便理解，我们拆成三个头
        self.fc_pi = nn.Linear(hidden_size, num_gaussians * z_dim)      # 混合概率
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * z_dim)      # 均值
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * z_dim)   # 标准差

    def forward(self, z, action, hidden=None):
        """
        z: [Batch, Seq_Len, 32]
        action: [Batch, Seq_Len, 3]
        """
        # 1. 拼接输入
        # x shape: [Batch, Seq_Len, 35]
        x = torch.cat([z, action], dim=-1)
        
        # 2. LSTM 前向传播
        self.lstm.flatten_parameters()
        # output: [Batch, Seq_Len, 256]
        output, hidden = self.lstm(x, hidden)
        
        # 3. 预测分布参数
        # 我们把 Batch 和 Seq_Len 合并处理，最后再变回来
        output_flat = output.reshape(-1, self.hidden_size)
        
        # 计算参数
        pi = self.fc_pi(output_flat)       # [B*S, K*Z]
        mu = self.fc_mu(output_flat)       # [B*S, K*Z]
        sigma = self.fc_sigma(output_flat) # [B*S, K*Z]
        
        # 4. 变换形状以便计算 Loss
        # 目标形状: [Batch, Seq_Len, K, Z]
        B, S, _ = z.shape
        K = self.num_gaussians
        Z = self.z_dim
        
        pi = pi.view(B, S, K, Z)
        mu = mu.view(B, S, K, Z)
        sigma = sigma.view(B, S, K, Z)
        
        # 5. 激活函数处理
        # pi: 混合概率，必须用 Softmax 保证和为 1
        pi = F.softmax(pi, dim=2)
        
        # sigma: 标准差，必须是正数，用 exp
        sigma = torch.exp(sigma)
        
        return pi, mu, sigma, hidden

def mdn_loss_function(pi, mu, sigma, target):
    """
    计算 Negative Log Likelihood (NLL) Loss
    target: 真实的下一个 z [Batch, Seq_Len, Z]
    """
    # 目标需要扩展维度以匹配 Gaussian 数量
    # target: [B, S, Z] -> [B, S, 1, Z]
    target = target.unsqueeze(2)
    
    # 1. 构建正态分布
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    
    # 2. 计算目标值在每个高斯分布下的对数概率 (Log Probability)
    # log_prob: [B, S, K, Z]
    log_probs = m.log_prob(target)
    
    # 3. 加上混合权重 (Log Space 计算防止下溢)
    # weighted_log_prob = log(pi * prob) = log(pi) + log_prob
    # pi: [B, S, K, Z]
    weighted_log_probs = torch.log(pi + 1e-8) + log_probs
    
    # 4. Log-Sum-Exp 技巧 (计算混合后的总概率)
    # 对 K 维度 (高斯核) 求和
    # log_sum_exp: [B, S, Z]
    log_prob_sum = torch.logsumexp(weighted_log_probs, dim=2)
    
    # 5. 取负号 (最大化概率 = 最小化负对数概率)
    loss = -log_prob_sum.mean()        # 正确：直接对刚才算出的变量求均值    
    
    return loss


if __name__ == "__main__":
    # 测试代码
    B, S, Z, A = 2, 10, 32, 3 # Batch=2, Seq=10, Z=32, Action=3
    
    model = MDNRNN(z_dim=Z, action_dim=A)
    dummy_z = torch.randn(B, S, Z)
    dummy_a = torch.randn(B, S, A)
    
    pi, mu, sigma, _ = model(dummy_z, dummy_a)
    
    print(f"Input z: {dummy_z.shape}")
    print(f"Output pi: {pi.shape}")       # 应为 [2, 10, 5, 32]
    print(f"Output mu: {mu.shape}")       # 应为 [2, 10, 5, 32]
    print(f"Output sigma: {sigma.shape}") # 应为 [2, 10, 5, 32]
    
    # 测试 Loss
    dummy_target = torch.randn(B, S, Z)
    loss = mdn_loss_function(pi, mu, sigma, dummy_target)
    print(f"Loss value: {loss.item()}")
    print("✅ MDN-RNN Architecture Check Passed!")