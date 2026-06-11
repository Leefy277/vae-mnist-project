# 变分自编码器 (VAE) - MNIST 手写数字生成与隐空间可视化

本项目使用 PyTorch 实现变分自编码器 (Variational Autoencoder, VAE)，并在 MNIST 数据集上进行训练。项目不仅包含基础的图像重建和生成功能，还深入探索了 VAE 的隐空间 (Latent Space)，提供了高质量的 2D 流形可视化和 3D 隐空间分布渲染，非常适合用于学术汇报和展示。

## 环境依赖

推荐使用支持 CUDA 的 NVIDIA GPU (如 RTX 4060) 进行训练以获得最佳性能。

- Python >= 3.12
- PyTorch (推荐匹配 CUDA 12.8 环境)
- torchvision
- matplotlib
- scipy
- numpy

##  项目结构

项目包含两种实现方式：单文件基础版和模块化进阶版。两个版本均已集成 Loss 曲线绘制功能。

- `vae/main.py`：单文件基础版。包含了完整的 VAE 架构定义 (隐变量维度为20)、训练逻辑、生成/重建测试，并在程序结束时自动导出 20 个 Epoch 的训练损失曲线。
- `vae_mnist/`：模块化进阶版。将模型、训练和可视化解耦，专注于 2D/3D 隐空间的探索。
  - `model.py`：定义了 VAE 的核心网络架构，包含 Encoder、重参数化技巧 (Reparameterization) 和 Decoder。
  - `train.py`：负责加载数据、计算损失 (BCE + KLD) 并执行训练循环，结束后自动保存权重文件并导出 2D 降维训练的 Loss 曲线。
  - `visualize.py`：加载训练好的模型，在 2D 隐空间内均匀采样，生成具有连续渐变效果的 2D 数字流形图。
  - `visualize_3d_digits.py`：加载 3D VAE 权重，使用 `matplotlib` 渲染高清晰度的 3D 数字隐空间散点图，并同步展现 3D 训练的 Loss 收敛轨迹。
  
## 运行

### 方法一：运行单文件基础版
直接进入 `vae` 文件夹并运行主程序，它会自动下载数据集并开始 20 个 Epoch 的训练：
```bash
cd vae
python main.py
```
运行产出：

vae_mnist.pth：模型权重文件。

generated_digits.png：随机生成的数字大图。

reconstruction.png：原图与模型重建图的对比。

vae_loss_curve.png：基础版 VAE 的整个训练收敛曲线图。
### 方法二：运行模块化进阶版
进入 vae_mnist 文件夹，按照以下顺序执行：

1. 训练模型
```bash
cd vae_mnist
python train.py
```
训练过程中的生成样本会保存在 results/ 目录下，训练完成后保存权重文件，并且会在 results/ 中自动生成 training_loss_curve.png。

2. 2D隐空间流形可视化
 ```bash
 python visualize.py
 ```
 程序会自动读取高斯分布的置信区间，生成平滑过渡的 2D 数字流形图，并保存至 results/latent_space_manifold.png。

3. 3D 隐空间分布可视化
(注：运行此脚本前需确保已有训练好的 3D VAE 权重文件 vae_mnist_3d.pth)

```bash
python visualize_3d_digits.py
```
程序将渲染带有清晰数字标签的 3D 散点图，直观展示不同类别数字在三维隐空间中的聚类效果，高清原图将保存至 results/vae_3d_digit_text_manifold_clear.png。

