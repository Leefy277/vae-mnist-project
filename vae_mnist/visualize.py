import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from model import VAE
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_latent_space(model, n=20, digit_size=28):
    """
    在 2D 隐空间内均匀网格采样，画出对应的生成数字流形图 (非常适合放进 PPT 展示)
    """
    # 创建一块大画布 (20*28) x (20*28) 像素
    figure = np.zeros((digit_size * n, digit_size * n))

    # 在高斯分布的置信区间内均匀产生坐标值
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                # 构造隐变量坐标 z
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(DEVICE)
                # 送入解码器生成图像
                x_decoded = model.decode(z_sample).cpu().numpy()
                digit = x_decoded[0].reshape(digit_size, digit_size)

                # 填入画布的对应网格中
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.title("VAE 2D Latent Space Manifold")
    os.makedirs('results', exist_ok=True)
    # 保存用于报告或PPT
    plt.savefig('results/latent_space_manifold.png', bbox_inches='tight')
    plt.show()
    print("可视化流形图已保存至 results/latent_space_manifold.png")


if __name__ == "__main__":
    model = VAE(latent_dim=2).to(DEVICE)
    if os.path.exists('vae_mnist.pth'):
        model.load_state_dict(torch.load('vae_mnist.pth', map_location=DEVICE))
        plot_latent_space(model)
    else:
        print("未找到训练好的模型权重文件 vae_mnist.pth，请先运行 train.py")