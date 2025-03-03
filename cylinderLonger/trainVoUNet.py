import torch
import torch.nn.functional as F
import logging
import os
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint, write_to_csv
from tools.visualization import save_error, save_prediction
from tools.loss import max_aeLoss
from parsercylinder import parse_args
from dataset.cylinderLong import CylinderDatasetVoronoi1D
from models.VoronoiUnet import voronoiUNet
# 配置全局参数
args = parse_args()  # 假设已实现parse_args函数
args.arch = "voronoiUNet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建目录结构
file_in = f"random_{args.random}_numpoints_{args.num_points}"
file_mid = os.path.join(f"./experiment_log/{args.arch}", file_in)
ckpt_dir = os.path.join(file_mid, args.ckpt_pth)
fig_dir = os.path.join(file_mid, args.fig_pth)
result_dir = os.path.join(file_mid, args.result_pth)

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 初始化数据集
train_dataset = CylinderDatasetVoronoi1D(data_path=args.data_pth, train=True)
test_dataset = CylinderDatasetVoronoi1D(data_path=args.data_pth, train=False)
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型
model = voronoiUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# 训练状态初始化
best_loss = float('inf')
start_epoch = 0
checkpoint_path = os.path.join(ckpt_dir, "checkpoint_best.pth")

# 加载检查点
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")


def train():
    global best_loss, start_epoch  # 访问全局状态

    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        train_loss, train_maxae, train_num = 0.0, 0.0, 0
        pbar = tqdm.tqdm(trainloader, desc=f"Training Epoch {epoch}")

        for inputs, outputs in pbar:
            inputs, outputs = inputs.to(device), outputs.to(device)

            # 前向传播
            preds = model(inputs).squeeze(1)

            # 计算损失
            loss = F.l1_loss(preds, outputs)
            maxae = max_aeLoss(preds, outputs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_maxae += maxae.item() * batch_size
            train_num += batch_size

            pbar.set_postfix(loss=loss.item(), maxae=maxae.item())

        # 更新学习率
        scheduler.step()

        # 记录训练日志
        avg_train_loss = train_loss / train_num
        avg_train_maxae = train_maxae / train_num
        write_to_csv(os.path.join(result_dir, "train_log.csv"),
                     [epoch, avg_train_loss, avg_train_maxae],
                     header=["epoch", "loss", "maxae"])

        # 验证阶段
        if epoch % args.val_interval == 0:
            val_loss, val_maxae = validate()

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }, filename=checkpoint_path)

            # 记录验证日志
            write_to_csv(os.path.join(result_dir, "val_log.csv"),
                         [epoch, val_loss, val_maxae],
                         header=["epoch", "loss", "maxae"])


def validate():
    model.eval()
    total_loss = 0.0
    total_maxae = 0.0
    total_num = 0
    saved_samples = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(testloader, desc="Validating")
        for inputs, outputs in pbar:
            inputs, outputs = inputs.to(device), outputs.to(device)

            # 前向传播
            preds = model(inputs).squeeze(1)

            # 计算指标
            loss = F.l1_loss(preds, outputs)
            maxae = max_aeLoss(preds, outputs)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_maxae += maxae.item() * batch_size
            total_num += batch_size

            # 保存可视化结果
            if saved_samples < 50:
                preds_np = preds.cpu().numpy().reshape(-1, 112, 192)
                outputs_np = outputs.cpu().numpy().reshape(-1, 112, 192)

                for i in range(preds_np.shape[0]):
                    if saved_samples >= 50:
                        break

                    # 保存结果
                    save_prediction(outputs_np[i], os.path.join(fig_dir, f"sample{saved_samples}_true.png"))
                    save_prediction(preds_np[i], os.path.join(fig_dir, f"sample{saved_samples}_pred.png"))
                    save_error(np.abs(outputs_np[i] - preds_np[i]),
                               os.path.join(fig_dir, f"sample{saved_samples}_error.png"))

                    saved_samples += 1

            pbar.set_postfix(loss=loss.item(), maxae=maxae.item())

    # 打印验证结果
    avg_loss = total_loss / total_num
    avg_maxae = total_maxae / total_num
    print(f"\nValidation Results:")
    print(f"MAE: {avg_loss:.6f}")
    print(f"Max-AE: {avg_maxae:.6f}")
    print(f"Saved {min(saved_samples, 50)} visualization samples")

    return avg_loss, avg_maxae


if __name__ == "__main__":
    train()