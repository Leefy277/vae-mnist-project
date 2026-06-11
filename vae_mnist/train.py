import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VAE
import os
import matplotlib.pyplot as plt

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

# 2. 创建一个列表，用来存储每个 Epoch 的平均损失
loss_history = []


def loss_function(recon_x, x, mu, logvar):
    """ VAE 损失函数 = 重构损失 (BCE) + KL散度 (KLD) """
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
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

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

    # 3. 将当前 Epoch 的平均损失记录下来
    loss_history.append(avg_loss)


if __name__ == "__main__":
    print(f"正在运行在设备: {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train(epoch)

        # 每个 Epoch 结束后测试生成效果
        with torch.no_grad():
            sample = torch.randn(64, 2).to(DEVICE)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png')

    # 保存权重
    torch.save(model.state_dict(), 'vae_mnist.pth')
    print("模型训练完成，权重已保存为 vae_mnist.pth")

    # 4. 训练完成后，自动绘制并保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Total Loss (BCE + KLD)', color='purple', linewidth=2,
             marker='o')
    plt.title('VAE Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 保存曲线图到 results 目录
    loss_curve_path = 'results/training_loss_curve.png'
    plt.savefig(loss_curve_path, bbox_inches='tight', dpi=300)
    print(f"成功生成训练曲线图并保存至: {loss_curve_path}")
    plt.show()