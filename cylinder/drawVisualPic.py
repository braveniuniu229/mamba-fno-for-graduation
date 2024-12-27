import numpy as np
import matplotlib.pyplot as plt
import cmocean
import os
def save_flow_field_at_intervals(data_path, t_interval, save_dir):
    """
    每隔 t_interval 个时刻保存流场图像。

    参数:
    - fields: 流场数据，形状为 (n, h, w)，n是时刻数，h和w分别是流场的高度和宽度
    - t_interval: 时间间隔（int），每隔多少个时刻保存一张图
    - save_dir: 图像保存的上级目录（str）
    """
    # 获取流场的形状
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # 加载数据集

    fields = np.load(data_path)
    l, _ = fields.shape
    fields = fields.reshape((l,384,199))
    n, h, w = fields.shape

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每个时间间隔，保存对应的图像
    for t in range(0, n, t_interval):
        # 获取对应时刻的流场数据
        field_at_t = fields[t]

        # 创建坐标网格
        x, y = np.linspace(0, w / 100.0, w), np.linspace(h / 100.0, 0, h)
        x, y = np.meshgrid(x, y)

        # 绘制流场图
        plt.figure(figsize=(5, 8))
        plt.contourf(x, y, field_at_t, levels=100, cmap=cmocean.cm.balance)

        # 设置标题
        plt.title(f"Flow Field at Time Step: {t}")

        # 构建保存路径
        output_path = os.path.join(save_dir, f"flow_field_{t}.png")

        # 保存图像
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved: {output_path}")
if __name__ == "__main__":
    dataDir = '../data/cylinder.npy'
    saveDir = './example'
    save_flow_field_at_intervals(data_path=dataDir,save_dir=saveDir,t_interval=15)

