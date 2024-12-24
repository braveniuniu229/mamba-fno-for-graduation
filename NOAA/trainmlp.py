# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : cylinder_mlp.py
import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv
from models.mlp import MLP
from dataset.NOAAtemperature import NOAADatasetMLP
from parseNOAA import parse_args
# from tools.visualization import plot3x1
from tools.utils import cre
from tools.visualization import plot3x1,plot3x1forgif,generate_gif_from_data
from tools.loss import Max_aeLoss
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
dataset_train = NOAADatasetMLP(args.data_pth, train=True, train_ratio=0.8, random_points=args.random,
                                   num_points=args.num_points)
dataset_test = NOAADatasetMLP(args.data_pth, train=False, train_ratio=0.8, random_points=args.random,
                                  num_points=args.num_points)
trainloader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
testloader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

def train():
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}

    net = MLP(layers=[args.num_points, 35,1280 ,1500,64800]).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
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
                    save_checkpoint(epoch, net, optimizer, val_loss, is_best, ckpt_dir)
                    print("New checkpoint saved in {}".format(ckpt_dir))
                net.train()


            # Plotting
            # if epoch % args.plot_freq == 0:
            #     plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(112, 192).cpu().numpy(),
            #             file_name=args.fig_path + f'/epoch{epoch}.png')
            #


def test():
    # 加载模型
    maxaeLoss = Max_aeLoss()
    net = MLP(layers=[args.num_points, 128, 1280, 4800, 64800]).to(device)

    # 加载checkpoint
    checkpoint = torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("total parameters:", count_parameters(net))

    total_l1_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0

    fields_list = []
    pres_list = []

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Testing", leave=True, colour='white')
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)

            pre = net(inputs)

            # 计算损失
            l1_loss_value = F.l1_loss(pre, outputs).item() * inputs.size(0)
            maxae_loss_value = maxaeLoss(pre, outputs).item() * inputs.size(0)

            total_l1_loss += l1_loss_value
            total_maxae_loss += maxae_loss_value
            total_samples += inputs.size(0)

            # reshape outputs and predictions
            pre_reshaped = pre.view(31 , 384, 199)
            outputs_reshaped = outputs.view(31, 384, 199)



            for i in range(0, 31, 1):
                true_values = outputs_reshaped[i].cpu().numpy()
                predicted_values = pre_reshaped[i].cpu().numpy()

                # plot3x1(true_values, predicted_values, file_name=os.path.join(fig_dir, f'figure_{i}.png'))
                fields_list.append(true_values)
                pres_list.append(predicted_values)

            pbar.update(1)

        avg_l1_loss = total_l1_loss / total_samples
        avg_maxae_loss = total_maxae_loss / total_samples
        output_gif = os.path.join(fig_dir, 'output.gif')
        generate_gif_from_data(fields_list, pres_list, output_gif)
        print(f"GIF saved as {output_gif}")
        print(f'Average L1 Loss: {avg_l1_loss}, Average Max AE Loss: {avg_maxae_loss}')


if __name__ == '__main__':
    train()
    test()
