import torch
import torch.nn.functional as F
import logging
import pickle
import os
import tqdm
from models.gappypod import GappyPodWeight1D
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
# Argument parsing
args.arch = "GasppyMLP"
best_loss = float("inf")
file_in = f"random_{args.random}_numpoints_{args.num_points}"
file_mid = os.path.join(f"./experiment_log/{args.arch}", file_in)
file_mid_orgin = os.path.join(f"./experiment_log/SDbaseline", file_in)
ckpt_dir = os.path.join(file_mid, args.ckpt_pth)
ckpt_dir_origin = os.path.join(file_mid_orgin,args.ckpt_pth)
fig_dir = os.path.join(file_mid, args.fig_pth)
result_dir = os.path.join(file_mid, args.result_pth)

# 使用 os.makedirs 递归创建目录
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
device = torch.device("cuda")

with open('../data/Cy_Taira.pickle', 'rb') as f:
# Load GappyPod

    data = pickle.load(f)

            # Convert to numpy array if needed
    data_np = np.array(data)
    data_np = data_np.reshape(data_np.shape[0], data_np.shape[1] * data_np.shape[2])
    origin_data = data_np[0:4000,]
    gappy_pod = GappyPodWeight1D(data=origin_data, map_size=112*192, n_components=200, observe_weight=50)
    dataset_test = CylinderDatasetMLP(args.data_pth, train=False, train_ratio=0.8, random_points=args.random,
                                      num_points=args.num_points)
    testloader = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    def val():
        net = MLP(layers=[args.num_points, 35,40 ,112*192]).cuda()
        checkpoint_path = os.path.join(ckpt_dir_origin, 'checkpoint_best.pth')


        # Load the model checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return

        # Initialize validation loss variables
        total_l1_loss = 0.0
        total_maxae_loss = 0.0
        total_samples = 0
        saved_samples = 0  # 已保存样本计数器

        # 创建可视化目录
        os.makedirs(fig_dir, exist_ok=True)

        # Set model to evaluation mode
        net.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(testloader), desc="Testing", leave=True, colour='white')
            for batch_idx, (inputs, outputs) in enumerate(testloader):
                inputs = inputs.to(device)  # Ensure inputs are on the correct device (GPU or CPU)
                outputs = outputs.to(device)
                predictions = net(inputs)
                pres = gappy_pod.reconstruct(predictions, inputs, weight=torch.ones_like(predictions))
                pres = pres.to(device)
                l1_loss = F.l1_loss(pres, outputs)
                maxae_loss = max_aeLoss(pres, outputs)

                # 累加指标
                batch_size = inputs.size(0)
                total_l1_loss += l1_loss.item() * batch_size
                total_maxae_loss += maxae_loss.item() * batch_size
                total_samples += batch_size

                # 保存前50个样本的可视化结果
                if saved_samples < 50:
                    # 转换数据为numpy数组
                    pred_np = pres.cpu().numpy().reshape(-1, 112, 192)  # (batch, 112, 192)
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


if __name__ == "__main__":
    val()
