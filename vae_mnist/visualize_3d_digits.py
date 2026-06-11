import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==========================================
# 1. 设置设备与加载数据 (🔥 将 batch_size 降为 150，这是防拥挤的物理前提)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=True)

# ==========================================
# 2. 定义 3D VAE 模型结构
# ==========================================
class VAE3D(nn.Module):
    def __init__(self):
        super(VAE3D, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 3)
        self.fc_logvar = nn.Linear(400, 3)
        self.fc3 = nn.Linear(3, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

# 实例化模型并加载权重
model = VAE3D().to(device)
if os.path.exists("vae_mnist_3d.pth"):
    model.load_state_dict(torch.load("vae_mnist_3d.pth", map_location=device))
    print("🚀 成功加载本地 3D VAE 模型权重文件！")
model.eval()

# ==========================================
# 3. 核心修复：画布放大与渲染优化
# ==========================================
data, labels = next(iter(test_loader))
data_flat = data.view(-1, 784).to(device)

with torch.no_grad():
    # 拿到真实的 3D 坐标 (不需要再乘以 scale_factor 了)
    mu, _ = model.encode(data_flat)
    mu = mu.cpu().numpy()
    labels = labels.numpy()

# 🔥 核心修改 1：把物理画布尺寸开到最大 (宽20, 高15)
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111, projection='3d')

print("正在渲染大尺寸、高清晰度的 3D 数字流形...")

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(len(mu)):
    x, y, z = mu[i, 0], mu[i, 1], mu[i, 2]
    digit_label = str(labels[i])

    # 🔥 核心修改 2：大字号(16)，加粗，并加上85%的透明度防死黑
    ax.text(x, y, z, digit_label,
            color=colors[labels[i]],
            fontsize=16,
            fontweight='bold',
            alpha=0.85,
            ha='center', va='center')

# 紧凑贴合数据边界，不留多余空白导致内部图形缩小
ax.set_xlim(mu[:, 0].min(), mu[:, 0].max())
ax.set_ylim(mu[:, 1].min(), mu[:, 1].max())
ax.set_zlim(mu[:, 2].min(), mu[:, 2].max())

# 字体同样放大
ax.set_title("VAE 3D Latent Space - Clear & Dispersed", fontsize=20, pad=30)
ax.set_xlabel("X Axis (Latent Dim 1)", fontsize=14)
ax.set_ylabel("Y Axis (Latent Dim 2)", fontsize=14)
ax.set_zlabel("Z Axis (Latent Dim 3)", fontsize=14)

# 调整到一个有立体感的观察视角
ax.view_init(elev=25, azim=60)

os.makedirs('results', exist_ok=True)
# 🔥 核心修改 3：大幅提高保存图片的 DPI (像素密度)，让字无比清晰
plt.savefig('results/vae_3d_digit_text_manifold_clear.png', bbox_inches='tight', dpi=300)
print("🔥 修复成功！请查看 results 目录下的高清大图！")
plt.show()