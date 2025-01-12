# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 15:32
# @Author  : zhaoxiaoyu
# @File    : loss.py
import torch
import torch.nn as nn


class OHEM2d(torch.nn.Module):
    """
    Weighted Loss
    """
    def __init__(self, loss_fun, weight=None):
        super(OHEM2d, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def forward(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        if inputs.ndim == 4:
            min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        elif inputs.ndim == 3:
            min, max = min.reshape(diff.shape[0], 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1).expand(diff.shape)
        diff = 10.0 * (diff - min) / (max - min)
        return torch.mean(torch.abs(diff * (inputs - targets)))


def max_aeLoss(output, label):
    """
    Compute the Max Absolute Error Loss.

    Parameters:
    - output: Tensor of shape (batch, H, W), the model's output.
    - label: Tensor of shape (batch, H, W), the ground truth.

    Returns:
    - loss: Scalar tensor, the mean max absolute error across the batch.
    """
    # 确保输出和标签的形状一致
    assert output.shape == label.shape, f"Output shape {output.shape} and Label shape {label.shape} must match!"

    # 计算绝对误差
    abs_error = torch.abs(output - label)  # Shape: (batch, H, W)

    # 对每个样本取最大值 (H, W 的范围内)
    max_error_per_sample = torch.max(abs_error.view(abs_error.size(0), -1), dim=1)[0]  # Shape: (batch,)

    # 对批次中的样本取均值
    loss = torch.mean(max_error_per_sample)  # Scalar
    return loss
class Max_aeLoss(nn.Module):
    def __init__(self):
        super(Max_aeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        # 计算 L1 损失
        l1_loss_value = self.l1_loss(outputs, targets)

        # 计算最后一个维度上的绝对值最大差
        abs_diff = torch.abs(outputs - targets)
        max_abs_diff, _ = torch.max(abs_diff, dim=-1)

        # 对这些最大差取平均
        mean_max_abs_diff = torch.mean(max_abs_diff)

        # 将两个损失值加起来（或者你可以根据需要调整权重）
        total_loss = l1_loss_value + mean_max_abs_diff

        return total_loss




class Mixloss(nn.Module):
    def __init__(self, w):  # w：MAE和Max——AE的权重
        super(Mixloss, self).__init__()
        if not 0 <= w <= 1:
            raise ValueError("Weight w must be between 0 and 1.")
        self.l1_loss = nn.L1Loss()
        self.w = w

    def forward(self, outputs, targets):
        # 计算 L1 损失
        l1_loss_value = self.l1_loss(outputs, targets)

        # 计算最后一个维度上的绝对值最大差
        abs_diff = torch.abs(outputs - targets)
        max_abs_diff, _ = torch.max(abs_diff, dim=-1)

        # 对这些最大差取平均
        mean_max_abs_diff = torch.mean(max_abs_diff)

        # 将两个损失值加起来（或者你可以根据需要调整权重）
        total_loss = self.w * l1_loss_value + (1 - self.w) * mean_max_abs_diff

        return total_loss