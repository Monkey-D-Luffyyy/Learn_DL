import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

# 训练数据
x_train = torch.linspace(-2, 2, 200).unsqueeze(1)  # [200, 1]
y_train = x_train**2

# 不同隐藏层宽度，看拟合效果
widths = [2, 4, 8, 16]
fig, axes = plt.subplots(1, len(widths), figsize=(4.2 * len(widths), 3.5))

for ax, width in zip(axes, widths):
    model = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(3000):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在更大范围上画，看外推
    x_test = torch.linspace(-2.5, 2.5, 500).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(x_test)

    ax.plot(x_test.numpy(), (x_test**2).numpy(), label="y = x²", linewidth=2)
    ax.plot(x_test.numpy(), y_pred.numpy(), "--", label=f"hidden={width}")
    ax.set_title(f"隐藏层 {width} 个 ReLU")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
