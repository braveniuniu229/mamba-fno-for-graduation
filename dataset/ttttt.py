import numpy as np

X = 100  # 空间维度 X 的大小
Y = 100  # 空间维度 Y 的大小
n = 3  # 我们希望生成 n x n 个点

x = np.linspace(0, X, n)
y = np.linspace(0, Y, n)
xx, yy = np.meshgrid(x, y)
print(xx,yy)
print(xx.ravel().shape)
points_uniform = np.vstack([xx.ravel(), yy.ravel()]).T
print(points_uniform.shape)
grid_x1, grid_y1 = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 63, 64))

# 使用 range
grid_x2, grid_y2 = np.meshgrid(range(64), range(64))

# 使用 arange
grid_x3, grid_y3 = np.meshgrid(np.arange(64), np.arange(64))

print("grid_x1 (linspace):\n", grid_x1)
print("grid_x2 (range):\n", grid_x2)
print("grid_x3 (arange):\n", grid_x3)