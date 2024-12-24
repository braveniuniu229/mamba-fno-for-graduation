import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint, count_parameters, write_to_csv, save_args
from dataset.cylinderdataset import CylinderDatasetLSTM
from parsercylinder import parse_args
from tools.utils import cre
from tools.visualization import plot3x1
from tools.loss import Max_aeLoss
from models.lstm import LSTMModel

# 配置参数
args = parse_args()
args.arch = "LSTM_Model_classic_train"
args.d_model = args.num_points
args.d_model_out = 76416


print(args)
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
# 保存训练参数到ckpt
save_args(args, os.path.join(ckpt_dir, "args.json"))
device = torch.device("cuda")

train_dataset = CylinderDatasetLSTM(data_path=args.data_pth, train=True)

test_dataset = CylinderDatasetLSTM(data_path=args.data_pth, train=False)
trainloader = DataLoader(train_dataset, collate_fn=None)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train():
    global best_loss
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')

    net = LSTMModel(input_size=args.num_points, hidden_size=128, output_size=76416, num_layers=2).to(device)
    start_epoch = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
    print("total parameters:", count_parameters(net))
    for epoch in range(start_epoch, args.epochs):
        # 训练过程
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

        # 验证过程
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

def test():
    # 加载模型
    maxaeLoss = Max_aeLoss()
    net = LSTMModel(input_size=args.num_points, hidden_size=128, output_size=76416, num_layers=2).to(device)

    # 加载checkpoint
    checkpoint = torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Testing", leave=True, colour='white')
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)

            pre = net(inputs)
            l1_loss_value = F.l1_loss(pre, outputs).item() * inputs.size(0)
            maxae_loss_value = maxaeLoss(pre, outputs).item() * inputs.size(0)

            total_l1_loss += l1_loss_value
            total_maxae_loss += maxae_loss_value
            total_samples += inputs.size(0)

            # reshape outputs and predictions
            pre_reshaped = pre.view(inputs.size(0), 31, 384, 199)
            outputs_reshaped = outputs.view(inputs.size(0), 31, 384, 199)

            for j in range(inputs.size(0)):
                for i in range(0, 31, 5):
                    true_values = outputs_reshaped[j, i].cpu().numpy()
                    predicted_values = pre_reshaped[j, i].cpu().numpy()

                    plot3x1(true_values, predicted_values, file_name=os.path.join(fig_dir, f'figure_{j}_{i}.png'))

            pbar.update(1)

    avg_l1_loss = total_l1_loss / total_samples
    avg_maxae_loss = total_maxae_loss / total_samples

    print(f'Average L1 Loss: {avg_l1_loss}, Average Max AE Loss: {avg_maxae_loss}')

if __name__ == '__main__':
    train()
    print("best val loss{}".format(best_loss))
    test()
