import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CylinderDatasetMLP(Dataset):
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
        self.num_train = 100
        self.num_test = self.length - self.num_train

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
        self.num_train = 100


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


class CylinderDatasetLSTMBeta(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, random_points=False, num_points=16,
                 slice_lengths=[2, 5, 10, 20, 30, 50]):
        """
        Custom dataset initializer.
        :param data_path: Path to the data (e.g., 'T' from your dataset)
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        :param random_points: Boolean flag indicating whether to select points randomly. Default is False.
        :param num_points: Number of points to select in space. Default is 16.
        :param slice_lengths: List of lengths to slice the sequences. Default is [2, 5, 10, 20, 30, 50].
        """

        self.data = np.load(data_path)
        self.train = train
        self.slice_lengths = slice_lengths

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
        input = self.data[start_idx:start_idx + length, self.points]
        output = self.data[start_idx:start_idx + length, :]
        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)


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


# Usage


# Example usage
if __name__ == "__main__":
    path = '../data/cylinder.npy'
    # train_dataset = CylinderDatasetLSTMBeta(data_path=path, train=True, slice_lengths=[2, 5, 10, 20, 30, 50,100,120])
    # train_sampler = SameLengthBatchSampler(train_dataset.slices, batch_size=32)
    # val_dataset = CylinderDatasetLSTM(data_path=path, train=False)
    # train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,collate_fn=None)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    t1 =CylinderDatasetMLP(data_path=path,train=True,random_points=False,num_points=16)
    t2 = CylinderDatasetMLP(data_path=path,train=False, random_points=False, num_points=16)
    train_loader =DataLoader(dataset=t1,batch_size=32)
    val_loader = DataLoader(dataset=t2, batch_size=32)



    for data, label in train_loader:
        print(data.shape, label.shape)
    print(1111)
    for data, label in val_loader:
        print(data.shape, label.shape)
    print(1111)

