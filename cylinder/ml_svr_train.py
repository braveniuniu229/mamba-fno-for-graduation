import numpy as np
import matplotlib.pyplot as plt
from models.traditionalML import svr_regression
import random
from tools.visualization import plot3x1
import os
from tools.visualization import save_error,save_prediction


# 数据加载和划分
def load_and_split_data(data):
    """
    划分数据为训练集和测试集。
    input_data: 输入数据 (形状: num_samples, 384*199)
    output_data: 输出数据 (形状: num_samples, 384*199)
    """
    # 训练集与测试集的划分（前100为训练集，后51为测试集）
    n = data.shape[1]
    selected_indices = np.linspace(0, n - 1, 16, dtype=int)
    train_inputs, val_inputs = data[:100,selected_indices], data[:,selected_indices]
    train_outputs, val_outputs = data[:100,], data

    return train_inputs, train_outputs, val_inputs, val_outputs


# 从384*199点中均匀选择16个点作为输入


# 计算损失（平均绝对误差和最大绝对误差）
def compute_losses(true_outputs, predicted_outputs):
    """
    计算所有样本的平均绝对误差（MAE）和最大绝对误差的平均值（MaxAE）。

    Args:
        true_outputs (numpy.ndarray): 真实输出 (num_samples, 384*199)
        predicted_outputs (numpy.ndarray): 预测输出 (num_samples, 384*199)

    Returns:
        mean_abs_error (float): 所有样本的平均绝对误差
        mean_max_abs_error (float): 每个样本的最大绝对误差的平均值
    """
    # 计算每个点的绝对误差
    abs_errors = np.abs(true_outputs - predicted_outputs)

    # 计算所有样本的平均绝对误差（标量）
    mean_abs_error = np.mean(abs_errors)

    # 计算每个样本的最大绝对误差
    max_abs_errors_per_sample = np.max(abs_errors, axis=1)

    # 计算最大绝对误差的平均值（标量）
    mean_max_abs_error = np.mean(max_abs_errors_per_sample)

    return mean_abs_error, mean_max_abs_error

# 绘制并保存误差图像（每个点的误差）
def plot_and_save_difference(true_output, predicted_output, sample_idx, output_dir="output_images"):
    """
    绘制真实输出与预测输出的差异图像，并保存图片。
    """
    # 计算绝对误差
    abs_diff = np.abs(true_output - predicted_output).reshape(384, 199)

    # 设置cmoccean配色
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_diff, cmap='cmoccean.deep', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Absolute Error (Sample {sample_idx})")
    plt.savefig(f"{output_dir}/abs_diff_sample_{sample_idx}.png")
    plt.close()

def val():
    # Load data (validation data from the cylinder.npy file)
    data = np.load('../data/cylinder.npy')

    # Split data (train, val datasets)
    train_inputs, train_outputs, val_inputs, val_outputs = load_and_split_data(data)

    # Initialize SVR model and train
    svr_model = svr_regression(train_inputs, train_outputs, train_inputs, train_outputs)
    svr_predictions = svr_model.predict(val_inputs)

    print(svr_predictions.shape)  # Expected output: (51, 384*199)

    # Compute losses
    mean_abs_error_per_sample, max_abs_error_per_sample = compute_losses(val_outputs, svr_predictions)
    fig_pth = os.path.join("ml_svr","figs")
    os.makedirs(fig_pth,exist_ok=True)
    for i in range(20):
        truevalues = val_outputs[i].reshape(384, 199)
        predict = svr_predictions[i].reshape(384, 199)
        error_file_name = os.path.join(fig_pth, f'time_step{i}_error.png')
        predicted_file_name = os.path.join(fig_pth, f'time_step{i}_predicted.png')
        save_error(abs(truevalues - predict), error_file_name)
        save_prediction(predict, predicted_file_name)

    # Output the loss information
    print(f"Mean Absolute Error per Sample: {mean_abs_error_per_sample}")
    print(f"Max Absolute Error per Sample: {max_abs_error_per_sample}")

    # # Create directory to save images if it doesn't exist
    # output_dir = os.path.join("output_images", "svr_val")
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # Randomly select 5 samples to plot
    # random_sample_indices = random.sample(range(51), 5)
    # for idx in random_sample_indices:
    #     ground_truth = val_outputs[idx].reshape(384, 199)
    #     prediction = svr_predictions[idx].reshape(384, 199)
    #
    #     # Plot and save the error images
    #     plot_and_save_difference(ground_truth, prediction, idx, output_dir)
    #
    # print(f"Validation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    # # 假设你的数据已经以 (151, 384*199) 形式存在
    # data = np.load('../data/cylinder.npy')
    #
    # # 加载并划分数据
    # train_inputs, train_outputs, val_inputs, val_outputs = load_and_split_data(data)
    #
    # # 训练SVR模型
    # svr_model = svr_regression(train_inputs, train_outputs, train_inputs, train_outputs)
    # svr_predictions = svr_model.predict(val_inputs)
    #
    # print(svr_predictions.shape)  # 预计输出 (51, 384*199)
    #
    # # 计算损失
    # mean_abs_error_per_sample, max_abs_error_per_sample = compute_losses(val_outputs, svr_predictions)
    #
    # # 输出损失信息
    # print(f"Mean Absolute Error per Sample: {mean_abs_error_per_sample}")
    # print(f"Max Absolute Error per Sample: {max_abs_error_per_sample}")
    #
    # # 随机选择5个样本绘制误差图
    # random_sample_indices = random.sample(range(51), 5)
    # dir_name = "ml_svr"
    # os.makedirs(dir_name,exist_ok=True)
    # for idx in random_sample_indices:
    #     ground_truth = val_outputs[idx].reshape(384,199)
    #     prediction = svr_predictions[idx].reshape(384,199)
    #
    #     file_name = f"output_image_{idx}.png"
    #     save_dir = os.path.join(dir_name,file_name)
    #     plot3x1(ground_truth, prediction, save_dir)
    val()