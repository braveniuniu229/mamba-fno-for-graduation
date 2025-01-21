import torch
import torch.nn.functional as F
import os
import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.mlp import shallow_decoder
from dataset.cylinderdataset import CylinderDatasetMLP
from tools.visualization import save_error,save_prediction
from tools.loss import max_aeLoss
import numpy as np
import random
import matplotlib.pyplot as plt
# Argument parsing
def parse_args():
    parser = ArgumentParser(description="Training shallow_decoder model")
    parser.add_argument('--data_pth', type=str, default="../data/cylinder.npy")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=4000, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help="Learning rate decay rate")
    parser.add_argument('--weight_decay_rate', type=float, default=0.8, help="Weight decay rate")
    parser.add_argument('--lr_decay_epoch', type=int, default=100, help="Epoch interval for learning rate decay")
    parser.add_argument('--n_sensors', type=int, default=16, help="Number of input sensors")
    parser.add_argument('--output_size', type=int, default=76416, help="Output size of the model")
    parser.add_argument('--val_interval', type=int, default=5, help="Validation interval")
    parser.add_argument('--ckpt_pth', type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument('--fig_pth', type=str, default="figures", help="Path to save figures")
    parser.add_argument('--log_pth', type=str, default="shallowdecoder/logs", help="Path to save logs")
    return parser.parse_args()

# Exponential learning rate scheduler
def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, weight_decay_rate=0.8, lr_decay_epoch=100):
    if epoch % lr_decay_epoch:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_rate
        param_group['weight_decay'] *= weight_decay_rate

# Training function
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    os.makedirs(os.path.join("shallowdecoder", args.ckpt_pth), exist_ok=True)
    fig_pth = os.makedirs(os.path.join("shallowdecoder", args.fig_pth), exist_ok=True)
    os.makedirs(args.log_pth, exist_ok=True)

    # Dataset and DataLoader
    train_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=True)
    test_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=False)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, and scheduler
    model = shallow_decoder(outputlayer_size=args.output_size, n_sensors=args.n_sensors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    best_loss = float('inf')
    best_maeloss = float('inf')
    checkpoint_path = os.path.join("shallowdecoder", args.ckpt_pth, f"checkpoint.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Load checkpoint if available
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    # Log files
    train_log_path = os.path.join(args.log_pth, "train_logs.csv")
    val_log_path = os.path.join(args.log_pth, "val_logs.csv")
    with open(train_log_path, "w") as f:
        f.write("epoch,train_loss,train_maxae_loss\n")
    with open(val_log_path, "w") as f:
        f.write("epoch,val_loss,val_maxae_loss\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_maxae_loss, train_num = 0.0, 0.0, 0
        pbar = tqdm.tqdm(total=len(trainloader), desc=f"Training Epoch {epoch}", leave=True, colour='white')

        for inputs, outputs in trainloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            predictions = model(inputs)
            loss = F.l1_loss(predictions, outputs)
            maxaeloss = max_aeLoss(predictions, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_maxae_loss += maxaeloss.item() * inputs.size(0)
            train_num += inputs.size(0)
            pbar.set_postfix(loss=loss.item(), maxae_loss=maxaeloss.item())
            pbar.update(1)

        train_loss /= train_num
        train_maxae_loss /= train_num
        print(f"Epoch {epoch}, Training Loss: {train_loss}, Training MaxAE Loss: {train_maxae_loss}")

        # Log training metrics
        with open(train_log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_maxae_loss}\n")

        # Apply custom LR scheduler
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=args.lr_decay_rate,
                         weight_decay_rate=args.weight_decay_rate, lr_decay_epoch=args.lr_decay_epoch)

        # Validation
        if epoch % args.val_interval == 0:
            model.eval()
            val_loss, val_maxae_loss, val_num = 0.0, 0.0, 0
            with torch.no_grad():
                pbar = tqdm.tqdm(total=len(testloader), desc=f"Validation Epoch {epoch}", leave=True, colour='white')
                for inputs, outputs in testloader:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    predictions = model(inputs)
                    loss = F.l1_loss(predictions, outputs)
                    maxaeloss = max_aeLoss(predictions, outputs)

                    val_loss += loss.item() * inputs.size(0)
                    val_maxae_loss += maxaeloss.item() * inputs.size(0)
                    val_num += inputs.size(0)
                    pbar.set_postfix(loss=loss.item(), maxae_loss=maxaeloss.item())
                    pbar.update(1)

            val_loss /= val_num
            val_maxae_loss /= val_num
            print(f"Epoch {epoch}, Validation Loss: {val_loss}, Validation MaxAE Loss: {val_maxae_loss}")

            # Log validation metrics
            with open(val_log_path, "a") as f:
                f.write(f"{epoch},{val_loss},{val_maxae_loss}\n")

            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_maeloss = val_maxae_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'max-aeloss': val_maxae_loss
                }, checkpoint_path)
                print(f"New best model saved at {checkpoint_path}")

    print(f"Training completed. Best Validation Loss: {best_loss}, Best Validation MaxAE Loss: {best_maeloss}")


def val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load checkpoint
    model = shallow_decoder(outputlayer_size=args.output_size, n_sensors=args.n_sensors).to(device)
    checkpoint_path = os.path.join("shallowdecoder", args.ckpt_pth, "checkpoint.pth")

    # Load model checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Dataset and DataLoader for validation
    test_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize loss variables
    val_mae, val_maxae, val_num = 0.0, 0.0, 0

    model.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Validation", leave=True, colour='white')

        # For each batch, get the predicted and true values
        all_predictions = []
        all_outputs = []
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)

            # Get model predictions
            predictions = model(inputs)
            fig_pth = os.path.join("shallowdecoder",args.fig_pth)
            for i in range(20):
                truevalues = outputs[i].reshape(384, 199).cpu().numpy()
                predict = predictions[i].reshape(384, 199).cpu().numpy()
                error_file_name = os.path.join(fig_pth, f'time_step{i}_error.png')
                predicted_file_name = os.path.join(fig_pth, f'time_step{i}_predicted.png')
                save_error(abs(truevalues - predict), error_file_name)
                save_prediction(predict, predicted_file_name)
            #

            pbar.update(1)


        # Select the first 5 timesteps

def compute_avg_abs_error(testloader, model, device, h, w):
    abs_errors = np.zeros((h, w))  # 初始化一个大小为 (h, w) 的误差矩阵
    total_count = np.zeros((h, w))  # 记录每个点的出现次数

    model.eval()

    with torch.no_grad():
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            predictions = model(inputs)

            for idx in range(inputs.size(0)):  # 遍历 batch
                pred = predictions[idx].cpu().detach().numpy()  # 取出当前样本的预测值
                true = outputs[idx].cpu().detach().numpy()  # 取出当前样本的真实值

                # 计算每个点的绝对误差并更新
                for i in range(h):
                    for j in range(w):
                        index = i * w + j
                        abs_errors[i, j] += abs(pred[index] - true[index])  # 累加绝对误差
                        total_count[i, j] += 1  # 记录该点的出现次数

    # 计算每个点的平均绝对误差
    avg_abs_errors = abs_errors / total_count
    return avg_abs_errors


# 获取时序平均误差最大的五个点
def get_top_5_error_points(avg_abs_errors):
    # 找到五个最大误差的点
    flat_indices = np.argsort(avg_abs_errors.flatten())[-5:]  # 找到最大五个点的索引
    top_5_coords = [(index // avg_abs_errors.shape[1], index % avg_abs_errors.shape[1]) for index in flat_indices]
    return top_5_coords


# 测试函数
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    test_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型并加载检查点
    model = shallow_decoder(outputlayer_size=args.output_size, n_sensors=args.n_sensors).to(device)
    checkpoint_path = os.path.join("shallowdecoder", args.ckpt_pth, "checkpoint.pth")

    # 加载模型
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # 计算所有点的平均绝对误差
    h, w = 384, 199
    avg_abs_errors = compute_avg_abs_error(testloader, model, device, h, w)

    # 获取时序平均误差最大的 5 个点的坐标
    top_5_coords = get_top_5_error_points(avg_abs_errors)
    print(f"Top 5 error coordinates (h, w): {top_5_coords}")

    # 将这些坐标转化为平面上的索引
    indices = [i * w + j for i, j in top_5_coords]

    # 创建保存结果的数组
    predicted_values = []
    true_values = []

    # 在测试集上进行预测
    for inputs, outputs in testloader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        predictions = model(inputs)

        # 获取 batch 中五个点的预测值
        for idx in range(inputs.size(0)):  # 遍历 batch
            pred = predictions[idx].cpu().detach().numpy()  # 取出当前样本的预测值
            true = outputs[idx].cpu().detach().numpy()  # 取出当前样本的真实值

            # 提取五个点的值
            pred_values = [pred[index] for index in indices]
            true_values_for_sample = [true[index] for index in indices]

            predicted_values.append(pred_values)
            true_values.append(true_values_for_sample)

            # 打印或保存结果
            print(f"Batch {idx + 1}:")
            print("Predicted values at selected coordinates:", pred_values)
            print("True values at selected coordinates:", true_values_for_sample)

    # 将预测值和真实值保存到文件（例如：CSV格式）
    np.savetxt("shallowdecoder/SD_predicted_values.csv", predicted_values, delimiter=",")
    np.savetxt("shallowdecoder/true_values.csv", true_values, delimiter=",")

    print("Results saved to SD_predicted_values.csv and true_values.csv.")


# 读取CSV文件并返回数据
def read_csv(file_path):
    return np.loadtxt(file_path, delimiter=",")


# 绘制时序图
def plot_time_series(predicted_values, true_values, top_5_coords, time_steps=51):
    for i, coord in enumerate(top_5_coords):
        # 获取每个点的预测值和真实值
        pred_vals = predicted_values[:, i]  # 每列是一个点的时序数据
        true_vals = true_values[:, i]  # 每列是一个点的时序数据

        # 确保每个点的预测值和真实值的长度都为 time_steps（即 52）
        if len(pred_vals) != time_steps or len(true_vals) != time_steps:
            print(f"Warning: Data for point {coord} does not have the correct length")
            continue

        # 创建图形
        plt.figure(figsize=(10, 6))

        # 时序轴
        time = np.arange(time_steps)

        # 绘制真实值和预测值
        plt.plot(time, pred_vals, label="Predicted", marker='o', linestyle='-', color='r')  # 预测值
        plt.plot(time, true_vals, label="True", marker='x', linestyle='--', color='b')  # 真实值

        # 标记时序为0, 5, 10, 15, 20...的点
        plt.xticks(time[::5])  # 每5个时刻显示一个标记

        # 添加图形标签
        plt.title(f"Time Series for Point {coord} (h={coord[0]}, w={coord[1]})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

        # 显示图形
        plt.grid(True)
        plt.show()


# Main
def main():
    args = parse_args()
    val(args)


if __name__ == "__main__":
    # predicted_values = read_csv("shallowdecoder/SD_predicted_values.csv")
    # true_values = read_csv("shallowdecoder/true_values.csv")
    # top_5_coords = [(40, 50), (120, 100), (200, 150), (250, 170), (310, 180)]  # 示例坐标
    #
    # # 绘制时序图
    # plot_time_series(predicted_values, true_values, top_5_coords)
    args = parse_args()

    main()
# Main

