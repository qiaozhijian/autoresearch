"""
一次性数据准备与评估（分类实验）。
下载 MNIST 并提供 DataLoader 与固定评估函数。

用法:
    python prepare.py   # 下载数据并打印就绪信息

数据存放在 ~/.cache/autoresearch/mnist/
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 常量（固定，勿改）
# ---------------------------------------------------------------------------

INPUT_DIM = 784
NUM_CLASSES = 10
TIME_BUDGET = 300  # 训练时间预算（秒，5 分钟）

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "mnist")

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def _get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def make_dataloader(split, batch_size, shuffle=None):
    """
    返回 MNIST 的 DataLoader。
    split: "train" 或 "val"
    batch_size: 批大小
    shuffle: train 时 True，val 时 False；默认按 split 自动设置。
    """
    assert split in ("train", "val")
    train_flag = split == "train"
    if shuffle is None:
        shuffle = train_flag

    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=train_flag,
        download=True,
        transform=_get_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(torch.cuda.is_available()),
    )


# ---------------------------------------------------------------------------
# 评估（勿改 — 固定指标）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    """准确率：正确预测数 / 总样本数。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    """验证集平均交叉熵损失。"""
    model.eval()
    total_loss = 0.0
    total_n = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total_n += y.size(0)
    return total_loss / total_n if total_n > 0 else 0.0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"数据目录: {DATA_DIR}")
    # 触发下载
    datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=_get_transform())
    datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=_get_transform())
    print("Done! Ready to train.")
