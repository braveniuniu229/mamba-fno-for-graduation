import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv
from models.VoronoiUnet import voronoiUNet
from dataset.cylinderLong import CylinderDatasetVoronoi1D
from parsercylinder import parse_args
from tools.visualization import save_error,save_prediction
from tools.loss import max_aeLoss
import numpy as np

args = parse_args()
print(args)
args.arch = "VoronoiUNet"
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
dataset_train = CylinderDatasetVoronoi1D(args.data_pth, train=True, train_ratio=0.8, random_points=args.random,
                                   num_points=args.num_points)
dataset_test = CylinderDatasetVoronoi1D(args.data_pth, train=False, train_ratio=0.8, random_points=args.random,
                                  num_points=args.num_points)
trainloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
testloader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)


def train():
    global best_loss
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')

    # 初始化UNet模型（保持原始模型定义）
    net = voronoiUNet().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}")

    print("Total parameters:", count_parameters(net))

    for epoch in range(args.epochs):
        # 训练阶段
        net.train()
        train_loss, train_num = 0., 0.
        pbar = tqdm.tqdm(trainloader, desc=f"Training Epoch {epoch}")

        for inputs, outputs in pbar:
            # 数据形状处理 [batch, C, H, W]
            inputs = inputs.float().to(device)
            outputs = outputs.view(outputs.size(0), 1, 112, 192).float().to(device)

            # 前向传播
            preds = net(inputs)
            preds = preds.squeeze(1)
            outputs = outputs.squeeze(1)
            # 计算损失
            loss = F.l1_loss(preds, outputs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            train_loss += loss.item() * inputs.size(0)
            train_num += inputs.size(0)
            pbar.set_postfix(loss=loss.item())

        # 更新学习率
        scheduler.step()

        # 记录训练日志
        avg_loss = train_loss / train_num
        write_to_csv(f'{result_dir}/train_log.csv', epoch, avg_loss)
        logging.info(f"Epoch {epoch}: Train Loss {avg_loss:.4f}")

        # 验证阶段
        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num = 0., 0.
            with torch.no_grad():
                pbar = tqdm.tqdm(testloader, desc=f"Validating Epoch {epoch}")
                for inputs, outputs in pbar:
                    # 数据形状处理
                    inputs = inputs.float().to(device)
                    outputs = outputs.view(outputs.size(0), 1, 112, 192).float().to(device)

                    preds = net(inputs)
                    preds = preds.squeeze(1)
                    outputs = outputs.squeeze(1)
                    loss = F.l1_loss(preds, outputs)

                    val_loss += loss.item() * inputs.size(0)
                    val_num += inputs.size(0)
                    pbar.set_postfix(loss=loss.item())

            avg_val_loss = val_loss / val_num
            logging.info(f"Epoch {epoch}: Val Loss {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_loss:
                is_best = True
                best_loss = avg_val_loss
                save_checkpoint(epoch, net, optimizer, val_loss, is_best, ckpt_dir)
                print("New checkpoint saved in {}".format(ckpt_dir))
            net.train()

def val():
    # 初始化模型
    net = voronoiUNet().to(device)

    # 加载检查点
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # 验证逻辑
    total_l1 = 0.0
    total_maxae = 0.0
    total_samples = 0
    saved_samples = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(testloader, desc="Testing")
        for inputs, outputs in pbar:
            # 数据形状转换
            inputs = inputs.float().to(device)
            outputs = outputs.view(outputs.size(0), 1, 112, 192).float().to(device)

            # 前向传播
            preds = net(inputs)
            preds = preds.squeeze(1)
            outputs = outputs.squeeze(1)
            # 计算指标
            l1_loss = F.l1_loss(preds, outputs)
            maxae = max_aeLoss(preds, outputs)

            total_l1 += l1_loss.item() * inputs.size(0)
            total_maxae += maxae.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # 可视化保存
            if saved_samples < 50:
                pred_np = preds.squeeze(1).cpu().numpy()  # [batch, 112, 192]
                true_np = outputs.squeeze(1).cpu().numpy()

                for i in range(pred_np.shape[0]):
                    if saved_samples >= 50: break

                    save_prediction(true_np[i], os.path.join(fig_dir, f"sample{saved_samples}_true.png"))
                    save_prediction(pred_np[i], os.path.join(fig_dir, f"sample{saved_samples}_pred.png"))
                    save_error(np.abs(true_np[i]-pred_np[i]), os.path.join(fig_dir, f"sample{saved_samples}_error.png"))
                    saved_samples += 1

    # 输出结果
    avg_l1 = total_l1 / total_samples
    avg_maxae = total_maxae / total_samples
    print(f"\nFinal Results:")
    print(f"MAE: {avg_l1:.6f}")
    print(f"Max-AE: {avg_maxae:.6f}")
    print(f"Saved {saved_samples} visualization samples")

if __name__ == '__main__':
    train()
    val()
