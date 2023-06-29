import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from fade.util import *
from fade.data import MNIST


class Experiment:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dataset = MNIST()
        self.loader_train, _ = self._get_data_loader(
            batch_size=257, number_workers=4, ratio=0.9, distinct=True
        )
        self.loader_test, _ = self._get_data_loader(
            batch_size=257, number_workers=4, ratio=0.1, distinct=True
        )
        self.model = ConvModel2()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.get_parameters(self.model), lr=1)
        self.scheduler = LinearLR(
            self.optimizer, total_iters=5, start_factor=0.05, end_factor=0.01
        )

    def _get_data_loader(self, batch_size, number_workers, ratio, distinct=False):
        dataset = self.dataset.get_subset(ratio, distinct)
        return (
            InfiniteIterator(
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=number_workers,
                    shuffle=True,
                )
            ),
            math.ceil(len(dataset) / batch_size),
        )

    def _loss(self, i, o):
        _, o = o.max(dim=-1)
        return self.loss(self.model(i), o)

    def _accuracy(self, i, o):
        return torch.true_divide(
            (self.model(i).max(dim=-1)[1] == o.max(dim=-1)[1]).sum(), i.size()[0]
        )

    def get_parameters(self, model):
        params = []
        for name, param in model.named_parameters():
            params.append(param)
        return params

    def run(self):
        for i in range(20):
            print(f"epoch {i+1}")
            self.train_epoch(i)
            self.evaluate()

    def train_epoch(self, epoch):
        self.model.train()
        for i, o in self.loader_train:
            i, o = i.to(self.device), o.to(self.device)
            self.optimizer.zero_grad()
            loss = self._loss(i, o)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            accuracy = 0
            batchCount = 1
            for i, o in self.loader_test:
                i, o = i.to(self.device), o.to(self.device)
                loss = ((batchCount - 1) * loss + self._loss(i, o)) / batchCount
                accuracy = (
                    (batchCount - 1) * accuracy + self._accuracy(i, o)
                ) / batchCount
        self.model.train()
        loss = get_numeric(loss)
        accuracy = get_numeric(accuracy)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")
        return (loss, accuracy)


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        width = 17
        depth = 10
        self.m = nn.Sequential(nn.Linear(784, width), nn.ReLU())
        for i in range(depth):
            self.m[-1] = nn.Linear(width, width)
            self.m[-1] = nn.ReLU()
        self.m[-1] = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        result = self.m(x)
        return result


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        depth = 3
        self.m = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        for i in range(depth):
            self.m[-1] = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
            self.m[-1] = nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(4, stride=4, padding=2)
        self.linear1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.m(x)
        x = self.pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        return x


class ConvModel2(nn.Module):
    def __init__(self):
        super(ConvModel2, self).__init__()
        self.m = nn.ModuleDict()
        for i in "abc":
            c = 1 if i == "a" else 8
            self.m[i] = nn.Sequential(
                nn.Conv2d(c, 8, kernel_size=3, stride=1, padding=1), nn.ReLU()
            )
        self.pooling = torch.nn.MaxPool2d(4, stride=4, padding=2)
        self.linear1 = nn.Linear(512, 10)

    def _reduce(self, tensors):
        reduced = torch.sum(torch.stack(tensors), dim=0)
        if False:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced

    def forward(self, x):
        s = "0abcd"
        outputs = {"0": x}
        for i in range(len(s) - 2):
            inputs = [self.m[s[i + 1]](outputs[s[i]])]
            outputs[s[i + 1]] = self._reduce(inputs)
        x = outputs[s[-2]]
        x = self.pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        return x


class A(nn.Module):
    def __init__(self, settings):
        super(A, self).__init__()
        self.settings = settings
        self.b = B(self.settings["model"])

    def change(self):
        self.settings["model"]["channels"] = "A"

    def test(self):
        print(self.settings["model"]["channels"])


class B:
    def __init__(self, settings):
        self.settings = settings

    def change(self):
        self.settings["channels"] = "B"

    def test(self):
        print(self.settings["channels"])


if __name__ == "__main__":
    Experiment().run()
