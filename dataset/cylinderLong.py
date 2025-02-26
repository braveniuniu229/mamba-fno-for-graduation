import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class CylinderflowDatasetLSTMBeta(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, random_points=False, num_points=16,
                 slice_lengths=[2, 5, 10, 20, 30, 50]):
        """
        Custom dataset initializer.
        :param data_path: Path to the pickle data file
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        :param random_points: Boolean flag indicating whether to select points randomly. Default is False.
        :param num_points: Number of points to select in space. Default is 16.
        :param slice_lengths: List of lengths to slice the sequences. Default is [2, 5, 10, 20, 30, 50].
        """
        # Load data from pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Convert to numpy array if needed
        data_np = np.array(data)

        # Remove the last dimension (which is 1) and flatten the spatial dimensions
        # From (5000, 112, 192, 1) to (5000, 112*192)
        data_np = data_np.reshape(data_np.shape[0], data_np.shape[1] * data_np.shape[2])

        self.data = data_np
        self.train = train
        self.slice_lengths = slice_lengths

        # Determine number of spatial points
        num_spatial_points = self.data.shape[1]  # Now this is 112*192

        # Select points in space
        if random_points:
            indices = np.random.choice(num_spatial_points, num_points, replace=False)
        else:
            indices = np.linspace(0, num_spatial_points - 1, num_points, dtype=int)

        self.points = indices

        # Determine split sizes
        self.length = self.data.shape[0]
        self.num_train = int(train_ratio * self.length)

        if self.train:
            self.timeslide = np.arange(self.num_train)
        else:
            self.timeslide = np.arange(self.num_train, self.length)

        self.data = self.data[self.timeslide, :]

        if self.train:
            # Generate all possible slices for the training dataset
            self.slices = self._generate_slices()
        else:
            # For validation/testing, use the entire sequence
            self.slices = [(0, self.data.shape[0])]

    def _generate_slices(self):
        slices = []
        for length in self.slice_lengths:
            for start_idx in range(0, self.data.shape[0] - length + 1):
                slices.append((start_idx, length))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        start_idx, length = self.slices[idx]
        input_data = self.data[start_idx:start_idx + length, self.points]
        output_data = self.data[start_idx:start_idx + length, :]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)


class SameLengthBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, slices, batch_size):
        self.slices = slices
        self.batch_size = batch_size
        self.slice_groups = self._group_by_length()

    def _group_by_length(self):
        slice_groups = {}
        for i, (_, length) in enumerate(self.slices):
            if length not in slice_groups:
                slice_groups[length] = []
            slice_groups[length].append(i)
        return slice_groups

    def __iter__(self):
        for length, indices in self.slice_groups.items():
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.slice_groups.values())


class CylinderDatasetVoronoi1D(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, random_points=False, num_points=16):
        """
        Custom dataset initializer.
        :param data_path: Path to the pickle data file
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        :param random_points: Boolean flag indicating whether to select points randomly. Default is False.
        :param num_points: Number of points to select in space. Default is 16.
        """
        # Load data from pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Convert to numpy array if needed
        data_np = np.array(data)

        # Remove the last dimension (which is 1) and reshape to (5000, 112, 192)
        self.data = data_np.reshape(data_np.shape[0], data_np.shape[1], data_np.shape[2])

        self.train = train
        self.num_points = num_points

        # Get dimensions from the actual data
        _, height, width = self.data.shape  # height=112, width=192

        if not random_points:
            # Ensure points are within valid range
            num_points_sqrt = int(math.sqrt(num_points))
            x = np.linspace(0, width - 1, num_points_sqrt, dtype=int)
            y = np.linspace(0, height - 1, num_points_sqrt, dtype=int)
            xv, yv = np.meshgrid(x, y)
            self.points = np.vstack([xv.ravel(), yv.ravel()]).T
        else:
            # Random points within valid range
            all_coordinates = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)
            self.points = all_coordinates[np.random.choice(all_coordinates.shape[0], num_points, replace=False)]

        # Determine split sizes
        self.length = self.data.shape[0]
        self.num_train = int(self.length * train_ratio)

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

        # Create grid for interpolation
        height, width = labels.shape
        grid_x, grid_y = np.meshgrid(range(width), range(height))

        # Perform Voronoi interpolation
        input = griddata(self.points, points_values, (grid_x, grid_y), method='nearest')

        # Create mask indicating sensor locations
        mask = np.zeros_like(labels, dtype=np.float32)
        mask[self.points[:, 1], self.points[:, 0]] = 1  # Note the swapped indices

        # Expand dimensions for channel-first format
        input_exp = np.expand_dims(input, axis=0)
        mask_exp = np.expand_dims(mask, axis=0)

        # Concatenate mask and interpolated field
        final_input = np.concatenate([input_exp, mask_exp], axis=0)

        return torch.tensor(final_input, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class CylinderDatasetMLP(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, random_points=False, num_points=16):
        """
        Custom dataset initializer.
        :param data_path: Path to the pickle data file
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        :param random_points: Boolean flag indicating whether to select points randomly. Default is False.
        :param num_points: Number of points to select in space. Default is 16.
        """
        # Load data from pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Convert to numpy array if needed
        data_np = np.array(data)

        # Remove the last dimension (which is 1) and flatten the spatial dimensions
        # From (5000, 112, 192, 1) to (5000, 112*192)
        self.data = data_np.reshape(data_np.shape[0], data_np.shape[1] * data_np.shape[2])

        self.train = train

        # Determine number of spatial points
        num_spatial_points = self.data.shape[1]  # Now this is 112*192

        # Select points in space
        if random_points:
            indices = np.random.choice(num_spatial_points, num_points, replace=False)
        else:
            indices = np.linspace(0, num_spatial_points - 1, num_points, dtype=int)

        self.points = indices

        # Determine split sizes
        self.length = self.data.shape[0]
        self.num_train = int(self.length * train_ratio)

        if self.train:
            self.timeslide = np.arange(self.num_train)
        else:
            self.timeslide = np.arange(self.num_train, self.length)

    def __len__(self):
        return len(self.timeslide)

    def __getitem__(self, idx):
        t_idx = self.timeslide[idx]
        input = self.data[t_idx, self.points]
        output = self.data[t_idx]
        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)
if __name__ == "__main__":
    dataset = CylinderflowDatasetLSTMBeta('../data/Cy_Taira.pickle', train=True)

# 创建批次采样器
    batch_sampler = SameLengthBatchSampler(dataset.slices, batch_size=32)

# 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
    for x,y in dataloader:
        print(x.shape,y.shape)