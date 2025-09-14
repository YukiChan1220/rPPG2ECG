from train.model import SimpleConvAE
import torch
import device

model = SimpleConvAE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    model.train()
    for xr, ye in train_loader:
        xr, ye = xr.to(device), ye.to(device)
        pred = model(xr)
        loss = criterion(pred, ye)
        opt.zero_grad(); loss.backward(); opt.step()
    # 验证 + 可视化若干样本
