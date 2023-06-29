import os
import numpy
import math
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from .util import Constants
from .persistence import *


class AbstractDataset(Dataset):
    def __init__ (self):
        super(AbstractDataset, self).__init__()

    def get_number_channels (self):
        return self.number_channels

    def get_number_features (self):
        return self.number_features

    def get_number_classes (self):
        return self.number_classes

    def get_sample(self):
        raise NotImplementedError()


class MyDataset(AbstractDataset):
    def __init__(self, size=100, number_features=20, overwrite=False):
        super(MyDataset, self).__init__()
        self.path = path([dir_data, "MyDataset"])
        try:
            os.makedirs(self.path)
        except:
            pass
        self.file_input = "input.npy"
        self.file_output = "output.npy"
        self.number_classes = 2
        self.number_channels = 1
        self.size = size
        self.number_features = number_features
        self.num_workers = 4
        if overwrite:
            try:
                self.delete_data()
            except:
                pass
        self.create_data()
        self.inputs, self.outputs = self.read_data()
        self.samples = list(zip(self.inputs, self.outputs))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def create_data(self):
        file = self.path / self.file_input
        if file.is_file():
            return
        else:
            inputs = numpy.array([numpy.around(numpy.random.rand(self.number_features)) for i \
                                  in range(self.size)], dtype=numpy.float32)
            outputs = numpy.array([self.calculate_output(i) for i in inputs])
            numpy.save(self.path / self.file_input, inputs)
            numpy.save(self.path / self.file_output, outputs)

    def calculate_output(self, i):
        input_size = len(i)
        criteria = int(numpy.sum(i[int(input_size/4):int(input_size/2)]) >= numpy.sum(i[int(input_size/2):int(3*input_size/4)]))
        return numpy.array([criteria, 1 - criteria], dtype=numpy.float32)
    
    def read_data(self):
        return self.read_file(self.path / self.file_input), self.read_file(self.path / self.file_output)
    
    def read_file(self, path):
        return numpy.load(path)
    
    def delete_data(self):
        """ not compatible with kaggle """
        (self.path / self.file_input).unlink()
        (self.path / self.file_output).unlink()

    def get_sample(self):
        loader_sample = DataLoader(
            torch.utils.data.Subset(self, [0]),
            batch_size = 1,
            num_workers = 0,
            shuffle=False
        )
        for i, o in loader_sample:
            return (i, o)
        
    def get_train_test_loader(self, batchSize):
        self.number_classes = 2
        self.train_test_ratio = 0.8
        loader_train = DataLoader(
                self,
                sampler=SubsetRandomSampler(range(0, int(self.train_test_ratio * self.size))),
                batch_size=batchSize,
                num_workers=self.num_workers
            )
        loader_test = DataLoader(
                self,
                sampler=SubsetRandomSampler(range(int(self.train_test_ratio * self.size), self.size)),
                batch_size=batchSize,
                num_workers=self.num_workers
            )
        return (loader_train, loader_test)


class MNIST(AbstractDataset):
    def __init__(self, one_hot_target=True):
        super(MNIST, self).__init__()
        self.path = Constants.path_mnist
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.number_classes = 10
        self.number_features = 28 * 28
        self.number_channels = 1
        self.one_hot_target = one_hot_target
        self.dataset = ConcatDataset([self._create(train=True), self._create(train=False)])
        self.ratio_distinct_total = .0

    def __len__ (self):
        return len(self.dataset)

    def _create (self, train):
        return torchvision.datasets.MNIST(
            self.path,
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
            target_transform=MyTargetTransform(self.one_hot_target)
        )

    def get_sample(self):
        return self.dataset[0]

    def get_subset (self, ratio, distinct):
        ratio_start = self.ratio_distinct_total if distinct else 0
        assert(ratio + ratio_start <= 1 or math.isclose(ratio + ratio_start, 1))
        self.ratio_distinct_total = self.ratio_distinct_total + ratio if distinct else self.ratio_distinct_total
        return torch.utils.data.Subset(self.dataset, range(math.ceil(ratio_start * len(self)), math.ceil((ratio_start + ratio) * len(self))))

    def get_sample_deprecated(self):
        # Todo: refactor
        loader_sample = DataLoader(
            torch.utils.data.Subset(self.dataset_test, [0]),
            batch_size = 1,
            num_workers = 0,
            shuffle=False
        )
        for i, o in loader_sample:
            return (i, o)


class MyTargetTransform:
    def __init__ (self, one_hot_target):
        self.one_hot_target = one_hot_target

    def __call__(self, x):
        t = torch.zeros(10)
        t[x] = 1
        if self.one_hot_target:
            return t
        else:
            return x


if __name__ == "__main__":
    m = MNIST()
    print(len(m))
    l = m.get_subset(0.5, False)
    print(len(l))
    b = m.get_subset(0.5, True)
    print(len(b))
    b2 = m.get_subset(0.25, True)
    print(len(b2))
    b3 = m.get_subset(0.25, True)
    print(len(b3))