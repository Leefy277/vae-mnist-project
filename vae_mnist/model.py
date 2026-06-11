import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        """
        变分自编码器 (VAE) 核心架构
        :param input_dim: 输入维度 (MNIST图片为 28x28 = 784)
        :param hidden_dim: 隐藏层维度
        :param latent_dim: 隐空间维度 (设为2便于在PPT中做2D流形可视化)
        """
        super(VAE, self).__init__()

        # 编码器 (Encoder) 架构：将图片压缩为分布参数
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # 预测隐变量的均值 mu
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # 预测隐变量方差的对数 log_var

        # 解码器 (Decoder) 架构：从隐空间重构图像
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """ 编码阶段：将输入映射到隐空间的分布参数 (mu, logvar) """
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        """ 
        重参数化技巧 (Reparameterization Trick) —— 答辩必问考点！
        z = mu + std * epsilon, 其中 epsilon 来自标准正态分布
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 从 N(0, I) 采样噪声，保证梯度可导
        return mu + eps * std

    def decode(self, z):
        """ 解码阶段：从隐空间变量 z 重构出原始输入 """
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 使用 Sigmoid 将输出像素限制在 [0, 1] 区间

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar