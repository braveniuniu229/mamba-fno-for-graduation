# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 21:41
# @Author  : zhaoxiaoyu
# @File    : mlp.py
import torch
import torch.nn as nn


class shallow_decoder(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(shallow_decoder, self).__init__()

        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size

        self.learn_features = nn.Sequential(
            nn.Linear(n_sensors, 40),
            nn.ReLU(True),
            nn.BatchNorm1d(40),
        )

        self.learn_coef = nn.Sequential(
            nn.Linear(40, 45),
            nn.ReLU(True),
            nn.BatchNorm1d(45),
        )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(45, self.outputlayer_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, x):
        x = self.learn_features(x)
        x = self.learn_coef(x)
        x = self.learn_dictionary(x)
        return x

class MLP(nn.Module):
    def __init__(self, layers=[4, 64, 128]):
        super(MLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            linear_layers.append(nn.ReLU())
        # linear_layers.append(nn.Dropout(0.1))
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.layers(x)


class PolyMLP(nn.Module):
    def __init__(self, layers=[4, 64, 128]):
        super(PolyMLP, self).__init__()
        linear_layers = []
        inject_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                nn.GELU()
            ))
            inject_layers.append(nn.Sequential(
                nn.Linear(layers[0], layers[i + 1]),
                nn.GELU()
            ))
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.ModuleList(linear_layers)
        self.inject_layers = nn.ModuleList(inject_layers)

    def forward(self, x):
        x_in = x
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x) * self.inject_layers[i](x_in)
        return self.layers[-1](x)


if __name__ == '__main__':
    net = shallow_decoder(n_sensors=16,outputlayer_size=384*199)

    x = torch.randn(5, 16)
    print(net(x).shape)
