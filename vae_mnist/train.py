import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VAE
import os

# 创建结果保存目录
os.makedirs('results', exist_ok=True)

# 超参数配置
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动下载与加载 MNIST 手写数字数据集
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型与优化器
model = VAE(input_dim=784, hidden_dim=400, latent_dim=2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def loss_function(recon_x, x, mu, logvar):
    """
    VAE 损失函数 = 重构损失 (BCE) + KL散度 (KLD)
    """
    # 1. 重构损失：衡量生成的图片与原图在像素级别上的差距
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 2. KL散度：衡量隐空间预测分布与标准正态分布 N(0, I) 的距离（起到正则化作用）
    # 由数学公式推导化简而来
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        # 前向传播
        recon_batch, mu, logvar = model(data)

        # 计算总损失
        loss = loss_function(recon_batch, data, mu, logvar)

        # 反向传播与梯度更新
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch [{epoch}/{EPOCHS}] Average Loss: {train_loss / len(train_loader.dataset):.4f}")


if __name__ == "__main__":
    print(f"正在运行在设备: {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train(epoch)

        # 每个 Epoch 结束后，在 2D 隐空间随机采样 64 个点，测试生成效果
        with torch.no_grad():
            sample = torch.randn(64, 2).to(DEVICE)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png')

    # 保存权重用于后续可视化
    torch.save(model.state_dict(), 'vae_mnist.pth')
    print("模型训练完成，权重已保存为 vae_mnist.pth")