# Learn_DL

深度学习练习、MNIST 训练，以及 LoRA rank 消融和 DPO 偏好对齐复现实验。

## 在 GPU 服务器上拉取

仓库是公开的，可以直接从 GPU 服务器拉取：

```bash
git clone git@github.com:Monkey-D-Luffyyy/Learn_DL.git
cd Learn_DL
```

如果服务器无法访问 GitHub 的 22 端口，也可以改用 HTTPS：

```bash
git clone https://github.com/Monkey-D-Luffyyy/Learn_DL.git
cd Learn_DL
```

仓库只保存压缩后的 MNIST 数据。首次拉取后解压：

```bash
unzip mnist_images.zip
```

解压后应出现 `mnist_images/train/` 和 `mnist_images/test/`。

## 环境安装

先检查显卡驱动与 CUDA：

```bash
nvidia-smi
python --version
```

建议使用 Python 3.10 或 3.11 的独立 Conda 环境：

```bash
conda create -n learn_dl python=3.10 -y
conda activate learn_dl
```

根据服务器驱动安装匹配的 PyTorch。例如 CUDA 12.6：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

安装后确认 PyTorch 能识别 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

国内网络下载 Hugging Face 模型时，可以设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 运行

MNIST 训练：

```bash
python train.py
```

LoRA rank 消融：

```bash
python repro_lora_dpo.py --exp lora
```

DPO 偏好对齐（可分别运行 beta=0.1、0.3 做消融）：

```bash
python repro_lora_dpo.py --exp dpo --beta 0.1
python repro_lora_dpo.py --exp dpo --beta 0.3
```

实验输出会写入 `out_lora_r*/`、`out_dpo_beta*/` 和 JSON 结果文件；这些运行产物默认不进入 Git。
