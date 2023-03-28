import os

import torch
import numpy as np
torch.manual_seed(100)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Conv2dWS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return nn.functional.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            Conv2dWS(in_channels=1, out_channels=2, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(14*14*2, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Получаем предсказания модели для данного батча
        pred = model(X)
        # Вычисляем лосс
        loss = loss_fn(pred, y)

        # Backpropagation
        # Обнуляем градиенты
        optimizer.zero_grad()
        # Вычисляем градиент лосса по параметрам модели
        loss.backward()
        # Производим шаг алгоритма оптимизации
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.detach().cpu().item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).detach().cpu().type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run_train(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(training_data, batch_size=64, sampler=sampler)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    print(f'Using {rank} device')
    model = NeuralNetwork().to(rank)
    if rank == 0:
        print(model)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # create model and move it to GPU with id rank
    ddp_model = DDP(model, device_ids=[rank])

    epochs = 3
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t+1}\n-------------------------------")
        ddp_model.train(True)
        train_loop(train_dataloader, ddp_model, loss_fn, optimizer, rank)
        if rank == 0:
            ddp_model.train(False)
            test_loop(test_dataloader, ddp_model, loss_fn, rank)
        torch.distributed.barrier()
    print("Done!")

    cleanup()

def run_demo(world_size):
    mp.spawn(run_train,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    gpus_available = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    run_demo(len(gpus_available))
