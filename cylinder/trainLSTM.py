import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint, count_parameters, write_to_csv, save_args
from dataset.cylinderdataset import CylinderDatasetLSTMBeta,SameLengthBatchSampler
from parsercylinder import parse_args
from tools.utils import cre
from tools.visualization import save_prediction,save_error
from tools.loss import max_aeLoss
from models.lstm import LSTMModel
import numpy as np

# 配置参数
args = parse_args()
args.arch = "LSTM_Model2"
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

train_dataset = CylinderDatasetLSTMBeta(data_path=args.data_pth, train=True, slice_lengths=[5])
train_sampler = SameLengthBatchSampler(train_dataset.slices, batch_size=32)
trainloader =  DataLoader(train_dataset, batch_sampler=train_sampler,collate_fn=None)
test_dataset = CylinderDatasetLSTMBeta(data_path=args.data_pth,train=False)
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
            maxae_loss_value = max_aeLoss(pre, outputs).item() * inputs.size(0)

            total_l1_loss += l1_loss_value
            total_maxae_loss += maxae_loss_value
            total_samples += inputs.size(0)

            # reshape outputs and predictions
            # pre_reshaped = pre.view(inputs.size(0), 31, 384, 199)
            # outputs_reshaped = outputs.view(inputs.size(0), 31, 384, 199)
            #
            # for j in range(inputs.size(0)):
            #     for i in range(0, 31, 5):
            #         true_values = outputs_reshaped[j, i].cpu().numpy()
            #         predicted_values = pre_reshaped[j, i].cpu().numpy()
            #
            #         plot3x1(true_values, predicted_values, file_name=os.path.join(fig_dir, f'figure_{j}_{i}.png'))

            pbar.update(1)

    avg_l1_loss = total_l1_loss / total_samples
    avg_maxae_loss = total_maxae_loss / total_samples

    print(f'Average L1 Loss: {avg_l1_loss}, Average Max AE Loss: {avg_maxae_loss}')
def val():
    # Initialize model
    net = LSTMModel(input_size=args.num_points, hidden_size=128, output_size=76416, num_layers=2).to(device)

    # Load the checkpoint
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Switch model to evaluation mode
    net.eval()

    total_l1_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0

    # Perform validation
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Validation", leave=True, colour='white')
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            outputs = outputs.squeeze(0)
            # Get predictions from the model
            pre = net(inputs)
            pre = pre.squeeze(0)
            # Calculate losses
            l1_loss_value = F.l1_loss(pre, outputs).item() * inputs.size(0)
            maxae_loss_value = max_aeLoss(pre, outputs).item() * inputs.size(0)

            # Accumulate the loss values
            total_l1_loss += l1_loss_value
            total_maxae_loss += maxae_loss_value
            total_samples += inputs.size(0)
            for i in range(20):
                truevalues = outputs[i].reshape(384,199).cpu().numpy()
                predict = pre[i].reshape(384,199).cpu().numpy()
                error_file_name = os.path.join(fig_dir, f'time_step{i}_error.png')
                predicted_file_name = os.path.join(fig_dir, f'time_step{i}_predicted.png')
                save_error(abs(truevalues-predict),error_file_name)
                save_prediction(predict,predicted_file_name)
            pbar.update(1)

    # Compute average loss values
    avg_l1_loss = total_l1_loss / total_samples
    avg_maxae_loss = total_maxae_loss / total_samples

    # Print the results
    print(f'Validation L1 Loss: {avg_l1_loss}')
    print(f'Validation MaxAE Loss: {avg_maxae_loss}')

# Save predicted values for five specific coordinates
def record(model, testloader, top_5_coords, device, file_name="lstm_predicted_values.csv"):
    predicted_values = []

    model.eval()
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)

            # Get model predictions
            predictions = model(inputs).squeeze(1).reshape(51, -1)  # 51 time steps, multiple points

            for coord in top_5_coords:
                i, j = coord
                point_pred_values = predictions[:, i * 199 + j].cpu().numpy()
                predicted_values.append(point_pred_values)

    np.savetxt(file_name, np.array(predicted_values).T, delimiter=",")
    print(f"Predicted values saved to {file_name}")

# Get five predefined coordinates
def get_top_5_coords():
    return [(1, 66), (1, 67), (0, 66), (0, 68), (0, 67)]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = LSTMModel(input_size=16, hidden_size=128, output_size=76416, num_layers=2).to(device)

    # Load the model checkpoint
    checkpoint_path = os.path.join("experiment_log", "LSTM_Model2", "random_False_numpoints_16", "ckpt", "checkpoint_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # DataLoader for testing
    test_dataset = CylinderDatasetLSTMBeta(data_path='../data/cylinder.npy', train=False)
    testloader = DataLoader(test_dataset, batch_size=51, shuffle=False)

    # Get the 5 predefined coordinates
    top_5_coords = get_top_5_coords()

    # Call the record function to save the predicted values to CSV
    record(model, testloader, top_5_coords, device, file_name="lstm_predicted_values.csv")


if __name__ == "__main__":
    val()
# if __name__ == '__main__':
#     val()
