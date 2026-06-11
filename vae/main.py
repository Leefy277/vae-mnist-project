# =====================
# 1. 导入库
# =====================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备：", device)


# =====================
# 3. 定义VAE模型
# =====================

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)

        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    # 编码器
    def encode(self, x):
        h = torch.relu(self.fc1(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    # 解码器
    def decode(self, z):
        h = torch.relu(self.fc3(z))

        return torch.sigmoid(self.fc4(h))

    # 前向传播
    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        return recon, mu, logvar


# =====================
# 4. 定义损失函数
# =====================

def loss_function(recon_x, x, mu, logvar):
    # 重建损失
    BCE = nn.functional.binary_cross_entropy(
        recon_x,
        x,
        reduction='sum'
    )

    # KL散度
    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return BCE + KLD


# =====================
# 5. 加载MNIST数据集
# =====================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# =====================
# 6. 创建模型
# =====================

model = VAE().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================
# 7. 训练函数
# =====================

def train(epoch):
    model.train()

    total_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.view(-1, 784).to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_function(
            recon_batch,
            data,
            mu,
            logvar
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch:{epoch} "
                f"Batch:{batch_idx} "
                f"Loss:{loss.item() / len(data):.4f}"
            )
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")


# =====================
# 8. 开始训练
# =====================

epochs = 20

for epoch in range(1, epochs + 1):
    train(epoch)

# =====================
# 9. 保存模型
# =====================

torch.save(
    model.state_dict(),
    "vae_mnist.pth"
)

print("模型保存成功")

# =====================
# 10. 生成新数字
# =====================

model.eval()

with torch.no_grad():
    z = torch.randn(64, 20).to(device)

    sample = model.decode(z)

    sample = sample.view(64,1,28,28)

    save_image(
        sample,
        "generated_digits.png",
        nrow=8
    )

print("生成图片已保存：generated_digits.png")

# =====================
# 11. 重建测试图片
# =====================

with torch.no_grad():
    data, _ = next(iter(train_loader))

    data = data[:8].view(-1, 784).to(device)

    recon, _, _ = model(data)

    recon = recon.view(8,1,28, 28)

    save_image(
        recon,
        "reconstruction.png",
        nrow=8
    )

print("重建图片已保存：reconstruction.png")

# =====================
# 12. 程序结束
# =====================

print("VAE训练完成")
