# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:10
# @Author  : zhaoxiaoyu
# @File    : visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
import cmocean
import io
import imageio

sbs.set_style('whitegrid')


def plot3x1(fields, pres, file_name):
    size = fields.shape
    x, y = np.linspace(0, size[1] / 100.0, size[1]), np.linspace(size[0] / 100.0, 0, size[0])
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(5, 8))
    plt.subplot(3, 1, 1)
    plt.contourf(x, y, fields, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.contourf(x, y, pres, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.contourf(x, y, pres - fields, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()

def plot3x1(fields, pres, file_name):
    size = fields.shape
    x, y = np.linspace(0, size[1] / 100.0, size[1]), np.linspace(size[0] / 100.0, 0, size[0])
    x, y = np.meshgrid(x, y)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

    cs1 = axes[0].contourf(x, y, fields, levels=100, cmap=cmocean.cm.balance)
    fig.colorbar(cs1, ax=axes[0])
    axes[0].set_title("Ground Truth")

    cs2 = axes[1].contourf(x, y, pres, levels=100, cmap=cmocean.cm.balance)
    fig.colorbar(cs2, ax=axes[1])
    axes[1].set_title("Prediction")

    cs3 = axes[2].contourf(x, y, pres - fields, levels=100, cmap=cmocean.cm.balance)
    fig.colorbar(cs3, ax=axes[2])
    axes[2].set_title("Error")

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_locations(positions, fields):
    """
    绘制测点位置
    :param positions: (n, 2) 包含n个测点的位置
    :param fields: 物理场
    :return:
    """
    h, w = fields.shape
    x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
    x_coor, y_coor = np.meshgrid(x_coor, y_coor)
    x_coor, y_coor = x_coor / 100.0, y_coor / 100.0

    x, y = [], []
    for i in range(positions.shape[0]):
        x.append(x_coor[positions[i, 0], positions[i, 1]])
        y.append(y_coor[positions[i, 0], positions[i, 1]])

    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap='jet')
    plt.figure(figsize=(9.6, 5.6))
    plt.axis('off')
    plt.pcolormesh(x_coor, y_coor, fields, cmap='seismic')
    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap=cmocean.cm.balance)
    plt.scatter(x, y, c='black')
    plt.show()


def plot_results(positions, fields,figpath):
    """
    绘制测点位置
    :param positions: (n, 2) 包含n个测点的位置
    :param fields: 物理场
    :return:
    """
    h, w = fields.shape
    x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
    x_coor, y_coor = np.meshgrid(x_coor, y_coor)
    x_coor, y_coor = x_coor / 100.0, y_coor / 100.0

    x, y = [], []
    for i in range(positions.shape[0]):
        x.append(x_coor[positions[i, 0], positions[i, 1]])
        y.append(y_coor[positions[i, 0], positions[i, 1]])

    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap='jet')
    plt.figure(figsize=(10.0, 5.0))
    plt.axis('off')
    plt.gca().set_aspect(1)
    # plt.pcolormesh(x_coor, y_coor, fields, cmap=cmocean.cm.balance)
    plt.contourf(x_coor, y_coor, fields, levels=100, cmap=cmocean.cm.balance)
    cbar = plt.colorbar()
    # C = plt.contour(x_coor, y_coor, fields, levels=[i * 0.5 + 15.5 for i in range(10)], colors="black", linewidths=0.5)
    # plt.clabel(C, inline=1, fontsize=7)
    # plt.clim(-11.5, 11.5)
    # cbar.formatter.set_powerlimits((0, 0))
    plt.scatter(x, y, c='black')
    plt.show()
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0, dpi=300)

def plot3x1forgif(fields, pres, time_step):
    size = fields.shape
    x, y = np.linspace(0, size[1] / 100.0, size[1]), np.linspace(size[0] / 100.0, 0, size[0])
    x, y = np.meshgrid(x, y)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

    cs1 = axes[0].contourf(x, y, fields, levels=100, cmap=cmocean.cm.balance)
    fig.colorbar(cs1, ax=axes[0])
    axes[0].set_title("Ground Truth")

    cs2 = axes[1].contourf(x, y, pres, levels=100, cmap=cmocean.cm.balance)
    fig.colorbar(cs2, ax=axes[1])
    axes[1].set_title("Prediction")

    cs3 = axes[2].contourf(x, y, pres - fields, levels=100, cmap=cmocean.cm.balance)
    fig.colorbar(cs3, ax=axes[2])
    axes[2].set_title("Error")

    plt.suptitle(f'Time Step: {time_step}', fontsize=16)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)

def generate_gif_from_data(fields_list, pres_list, output_gif):
    images = []
    time_steps = len(fields_list)

    for i in range(time_steps):
        image = plot3x1forgif(fields_list[i], pres_list[i], time_step=i)
        images.append(image)

    imageio.mimsave(output_gif, images, duration=0.5)



