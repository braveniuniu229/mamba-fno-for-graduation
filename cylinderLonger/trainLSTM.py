import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv
from models.lstm import LSTMModel
from dataset.cylinderLong import CylinderDatasetMLP
from parsercylinder import parse_args
from dataset.cylinderLong import CylinderflowDatasetLSTMBeta,SameLengthBatchSampler
from tools.utils import cre
from tools.visualization import save_error,save_prediction
from tools.loss import max_aeLoss
import numpy as np
# Configure the arguments
args = parse_args()
print(args)
args.arch = "LSTMbaseline"
best_loss = float("inf")
file_in = f"random_{args.random}_numpoints_{args.num_points}"
file_mid = os.path.join(f"./experiment_log/{args.arch}", file_in)
ckpt_dir = os.path.join(file_mid, args.ckpt_pth)
fig_dir = os.path.join(file_mid, args.fig_pth)
result_dir = os.path.join(file_mid, args.result_pth)

# 使用 os.makedirs 递归创建目录
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
device = torch.device("cuda")
train_dataset = CylinderflowDatasetLSTMBeta(data_path=args.data_pth, train=True, slice_lengths=[50])
train_sampler = SameLengthBatchSampler(train_dataset.slices, batch_size=32)
trainloader =  DataLoader(train_dataset, batch_sampler=train_sampler,collate_fn=None)
test_dataset = CylinderflowDatasetLSTMBeta(data_path=args.data_pth,train=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train():
    global best_loss
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')

    # 初始化LSTM模型
    net = LSTMModel(
        input_size=args.num_points,  # 输入特征维度
        hidden_size=128,  # LSTM隐藏层维度
        output_size=112 * 192,  # 输出特征维度
        num_layers=2  # LSTM层数
    ).to(device)

    start_epoch = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    print("Total parameters:", count_parameters(net))

    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        net.train()
        train_loss, train_num = 0., 0.
        pbar = tqdm.tqdm(total=len(trainloader), desc=f"Training Epoch {epoch}", colour='blue')

        for inputs, outputs in trainloader:
            # 数据转移 (保持原始形状 [batch, seq_len, input_dim])
            inputs = inputs.to(device, non_blocking=True)  # shape: [batch, 50, num_points]
            outputs = outputs.to(device, non_blocking=True)  # shape: [batch, 50, 112*192]

            # 前向传播
            predictions = net(inputs)  # output shape: [batch, 50, 112*192]

            # 计算损失（保持序列维度）
            loss = F.l1_loss(predictions, outputs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_num += batch_size

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        # 记录训练日志
        avg_train_loss = train_loss / train_num
        write_to_csv(f'{result_dir}/train_log.csv', epoch, avg_train_loss)
        scheduler.step()

        # 验证阶段
        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num = 0., 0.
            with torch.no_grad():
                pbar = tqdm.tqdm(total=len(testloader), desc=f"Validation Epoch {epoch}", colour='green')
                for inputs, outputs in testloader:
                    inputs = inputs.to(device)  # shape: [1, seq_len, num_points]
                    outputs = outputs.to(device)  # shape: [1, seq_len, 112*192]

                    predictions = net(inputs)
                    loss = F.l1_loss(predictions, outputs)

                    val_loss += loss.item() * inputs.size(0)
                    val_num += inputs.size(0)
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

                avg_val_loss = val_loss / val_num
                # 保存最佳模型
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    is_best = True
                    save_checkpoint(epoch, net, optimizer, val_loss, is_best, ckpt_dir)
                    print(f"New best checkpoint saved at {checkpoint_path}")

            net.train()


def val():
    # 初始化LSTM模型
    net = LSTMModel(
        input_size=args.num_points,
        hidden_size=128,
        output_size=112 * 192,
        num_layers=2
    ).to(device)

    # 加载检查点
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1 = 0.0
    total_maxae = 0.0
    total_samples = 0
    saved_samples = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(testloader, desc="Testing", colour='yellow')
        for inputs, outputs in pbar:
            inputs = inputs.to(device)  # [1, seq_len, num_points]
            outputs = outputs.to(device)  # [1, seq_len, 112*192]

            # 前向传播
            preds = net(inputs)  # [1, 50, 21504]

            # 计算指标（保持序列维度）
            l1_loss = F.l1_loss(preds, outputs)
            maxae = max_aeLoss(preds, outputs)

            total_l1 += l1_loss.item() * inputs.size(0)
            total_maxae += maxae.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # 可视化保存（每个样本保存完整序列）
            if saved_samples < 50:
                # 转换形状 [seq_len, 112, 192]
                pred_grid = preds.squeeze(0).view(-1, 112, 192).cpu().numpy()  # [50, 112, 192]
                true_grid = outputs.squeeze(0).view(-1, 112, 192).cpu().numpy()

                # 保存每个时间步
                for t in range(pred_grid.shape[0]):
                    save_prediction(true_grid[t], os.path.join(fig_dir, f"sample{saved_samples}_t{t}_true.png"))
                    save_prediction(pred_grid[t], os.path.join(fig_dir, f"sample{saved_samples}_t{t}_pred.png"))
                    save_error(np.abs(true_grid[t] - pred_grid[t]),
                               os.path.join(fig_dir, f"sample{saved_samples}_t{t}_error.png"))

                saved_samples += 1

    # 输出最终结果
    final_l1 = total_l1 / total_samples
    final_maxae = total_maxae / total_samples
    print(f"\nFinal Test Results:")
    print(f"MAE: {final_l1:.6f}")
    print(f"Max-AE: {final_maxae:.6f}")
    print(f"Visualized {saved_samples} samples in {fig_dir}")


if __name__ == '__main__':
    train()
    val()
