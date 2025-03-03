import torch
import torch.nn as nn


class VoronoiCNN(nn.Module):
    def __init__(self):
        super(VoronoiCNN, self).__init__()

        # 输入通道数2，输出通道数48，卷积核7x7，padding=3保持尺寸
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        # 最终输出层，3x3卷积核，padding=1保持尺寸
        self.final_conv = nn.Conv2d(48, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x

# 实例化模型
model = VoronoiCNN()
