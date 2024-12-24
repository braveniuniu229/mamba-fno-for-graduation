import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import griddata


class CylinderDatasetVoronoi1D(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, random_points=False, num_points=16):
        """
        Custom dataset initializer.
        :param data_path: Path to the data (e.g., 'T' from your dataset)
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        :param random_points: Boolean flag indicating whether to select points randomly. Default is False.
        :param num_points: Number of points to select in space. Default is 16.
        """

        data = np.load(data_path)
        self.data = data.reshape(151, 384, 199)  # Ensure data shape is (151, 384, 199)
        self.train = train
        self.num_points = num_points

        if not random_points:
            # Ensure points are within valid range
            num_points_sqrt = int(math.sqrt(num_points))
            x = np.linspace(0, 198, num_points_sqrt, dtype=int)
            y = np.linspace(0, 383, num_points_sqrt, dtype=int)
            xv, yv = np.meshgrid(x, y)
            self.points = np.vstack([xv.ravel(), yv.ravel()]).T
        else:
            # Random points within valid range
            all_coordinates = np.array(np.meshgrid(np.arange(199), np.arange(384))).T.reshape(-1, 2)
            self.points = all_coordinates[np.random.choice(all_coordinates.shape[0], num_points, replace=False)]

        # Determine split sizes
        self.length = self.data.shape[0]
        self.num_train = int(self.length * train_ratio)
        self.num_test = self.length - self.num_train

        if self.train:
            self.timeslide = np.arange(self.num_train)
        else:
            self.timeslide = np.arange(self.num_train, self.length)

    def __len__(self):
        return len(self.timeslide)

    def __getitem__(self, idx):
        t_idx = self.timeslide[idx]
        labels = self.data[t_idx]
        # Swap indices to match data shape
        points_values = labels[self.points[:, 1], self.points[:, 0]]  # Note the swapped indices

        grid_x, grid_y = np.meshgrid(range(199), range(384))
        input = griddata(self.points, points_values, (grid_x, grid_y), method='nearest')  # Voronoi

        mask = np.zeros_like(labels, dtype=np.float32)
        mask[self.points[:, 1], self.points[:, 0]] = 1  # Note the swapped indices

        input_exp = np.expand_dims(input, axis=0)
        mask_exp = np.expand_dims(mask, axis=0)

        # 将拓展后的mask和插值后的温度场拼接在一起
        final_input = np.concatenate([input_exp, mask_exp], axis=0)

        return torch.tensor(final_input, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class CylinderDatasetLSTM(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, random_points=False, num_points=16):
        """
        Custom dataset initializer.
        :param data_path: Path to the data (e.g., 'T' from your dataset)
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        :param random_points: Boolean flag indicating whether to select points randomly. Default is False.
        :param num_points: Number of points to select in space. Default is 16.
        """

        self.data = np.load(data_path)
        self.train = train

        # Determine number of spatial points
        num_spatial_points = self.data.shape[1]  # Assuming the second dimension is spatial

        # Select points in space
        if random_points:
            indices = np.random.choice(num_spatial_points, num_points, replace=False)
        else:
            indices = np.linspace(0, num_spatial_points - 1, num_points, dtype=int)

        self.points = indices

        # Determine split sizes
        self.length = self.data.shape[0]
        self.num_train = int(self.length * train_ratio)
        self.num_test = self.length - self.num_train

        if self.train:
            self.timeslide = np.arange(self.num_train)

        else:
            self.timeslide = np.arange(self.num_train, self.length)
        self.data = self.data[self.timeslide, :]

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        input = self.data[:,self.points]
        output = self.data
        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)

# Example usage
if __name__ == "__main__":
    path = '../data/cylinder.npy'
    dataset_train = CylinderDatasetVoronoi1D(path, train=True, train_ratio=0.8, random_points=False, num_points=16)
    dataset_test = CylinderDatasetVoronoi1D(path, train=False, train_ratio=0.8, random_points=True, num_points=16)


    trainloader = DataLoader(dataset_train, shuffle=True, batch_size=10)
    testloader = DataLoader(dataset_test,shuffle=False,batch_size=10)# Adjust batch size as needed
    for data, label in trainloader:
        print(data.shape, label.shape)
    print(11111)
    for data,label in testloader:
        print(data.shape,label.shape)
