import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv
from models.mlp import MLP
from dataset.cylinderLong import CylinderDatasetMLP
from parsercylinder import parse_args
from tools.utils import cre
from tools.visualization import save_error,save_prediction
from tools.loss import max_aeLoss
import numpy as np
# Configure the arguments
args = parse_args()
print(args)
args.arch = "SDbaseline"
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
dataset_train = CylinderDatasetMLP(args.data_pth, train=True, train_ratio=0.8, random_points=args.random,
                                   num_points=args.num_points)
dataset_test = CylinderDatasetMLP(args.data_pth, train=False, train_ratio=0.8, random_points=args.random,
                                  num_points=args.num_points)
trainloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
testloader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

def train():
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}

    net = MLP(layers=[args.num_points, 35,40 ,112*192]).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    print("total parameters:",count_parameters(net))
    for epoch in range(args.epochs):
        # Training procedure
        net.train()
        train_loss, train_num = 0., 0.
        pbar = tqdm.tqdm(total=len(trainloader), desc=f"Training Epoch {epoch}", leave=True, colour='white')
        for inputs, outputs in trainloader:
            inputs, outputs = inputs.cuda(non_blocking=True), outputs.cuda(non_blocking=True)
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.shape[0]
            train_num += inputs.shape[0]
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        train_loss = train_loss / train_num
        write_to_csv(f'{result_dir}/train_log.csv', epoch, train_loss)
        logging.info("Epoch: {}, Avg_loss: {}".format(epoch, train_loss))
        scheduler.step()

        # Validation procedure
        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num = 0., 0.
            with torch.no_grad():
                pbar = tqdm.tqdm(total=len(testloader), desc=f"Validation Epoch {epoch}", leave=True, colour='white')
                for inputs, outputs in testloader:
                    inputs, outputs = inputs.to(device), outputs.to(device)

                    pre = net(inputs)
                    loss = F.l1_loss(outputs, pre)

                    val_loss += loss.item() * inputs.shape[0]
                    val_num += inputs.shape[0]
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

                val_loss = val_loss / val_num
                logging.info("Epoch: {}, Val_loss: {}".format(epoch, val_loss))
                write_to_csv(f'{result_dir}/val_log.csv', epoch, val_loss)
                if val_loss < best_loss:
                    is_best = True
                    best_loss = val_loss
                    save_checkpoint(epoch, net, optimizer, val_loss, is_best, ckpt_dir)
                    print("New checkpoint saved in {}".format(ckpt_dir))
                net.train()


            # Plotting
            # if epoch % args.plot_freq == 0:
            #     plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(112, 192).cpu().numpy(),
            #             file_name=args.fig_path + f'/epoch{epoch}.png')
            #


def val():
    # 初始化模型
    net = MLP(layers=[args.num_points, 35, 40, 112 * 192]).cuda()

    # 加载最佳检查点
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0
    saved_samples = 0  # 已保存样本计数器

    # 创建可视化目录
    os.makedirs(fig_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Testing", leave=True, colour='white')
        for batch_idx, (inputs, outputs) in enumerate(testloader):
            # 转移数据到GPU
            inputs, outputs = inputs.cuda(), outputs.cuda()

            # 前向传播
            pred = net(inputs)

            # 计算损失
            l1_loss = F.l1_loss(pred, outputs)
            maxae_loss = max_aeLoss(pred, outputs)

            # 累加指标
            batch_size = inputs.size(0)
            total_l1_loss += l1_loss.item() * batch_size
            total_maxae_loss += maxae_loss.item() * batch_size
            total_samples += batch_size

            # 保存前50个样本的可视化结果
            if saved_samples < 50:
                # 转换数据为numpy数组
                pred_np = pred.cpu().numpy().reshape(-1, 112, 192)  # (batch, 112, 192)
                true_np = outputs.cpu().numpy().reshape(-1, 112, 192)

                # 保存当前batch内的样本
                for i in range(batch_size):
                    if saved_samples >= 50:
                        break

                    # 生成唯一文件名
                    sample_id = saved_samples

                    # 保存真实场
                    save_prediction(true_np[i],
                                    os.path.join(fig_dir, f"sample{sample_id}_true.png"))

                    # 保存预测场
                    save_prediction(pred_np[i],
                                    os.path.join(fig_dir, f"sample{sample_id}_pred.png"))

                    # 保存绝对误差场
                    error = np.abs(true_np[i] - pred_np[i])
                    save_error(error,
                               os.path.join(fig_dir, f"sample{sample_id}_error.png"))

                    saved_samples += 1

            pbar.update(1)
            pbar.set_postfix(l1_loss=l1_loss.item(), maxae_loss=maxae_loss.item())

    # 计算最终指标
    avg_l1 = total_l1_loss / total_samples
    avg_maxae = total_maxae_loss / total_samples

    print(f"\nTest Results:")
    print(f"MAE: {avg_l1:.6f}")
    print(f"Max-AE: {avg_maxae:.6f}")
    print(f"Saved visualizations for {saved_samples} samples to {fig_dir}")

    return avg_l1, avg_maxae


if __name__ == '__main__':
    train()
    val()
