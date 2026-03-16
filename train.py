"""
Autoresearch 分类脚本。单文件，固定时间预算。
使用 MNIST + 简单 MLP，目标为最高 val_accuracy。
用法: uv run train.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET,
    INPUT_DIM,
    NUM_CLASSES,
    make_dataloader,
    evaluate_accuracy,
    evaluate_loss,
)

# ---------------------------------------------------------------------------
# 模型（可改）
# ---------------------------------------------------------------------------

BATCH_SIZE = 128
HIDDEN = 512

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_DIM, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, NUM_CLASSES),
)

# ---------------------------------------------------------------------------
# 设备与种子
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

train_loader = make_dataloader("train", BATCH_SIZE)
val_loader = make_dataloader("val", BATCH_SIZE)

# ---------------------------------------------------------------------------
# 训练循环（固定时间预算）
# ---------------------------------------------------------------------------

t_start = time.time()
training_time = 0.0
step = 0
model.train()

while True:
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        step += 1

        t1 = time.time()
        training_time += t1 - t0
        if training_time >= TIME_BUDGET:
            break
        t0 = t1

    if training_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# 验证与输出
# ---------------------------------------------------------------------------

val_accuracy = evaluate_accuracy(model, val_loader, device)
val_loss = evaluate_loss(model, val_loader, device)
t_end = time.time()

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
num_params = sum(p.numel() for p in model.parameters())

print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"val_loss:        {val_loss:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:   {t_end - t_start:.1f}")
print(f"peak_vram_mb:    {peak_vram_mb:.1f}")
print(f"num_steps:       {step}")
print(f"num_params:      {num_params}")
