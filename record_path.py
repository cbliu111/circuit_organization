import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import time
from analyzer import vectorize_weights
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--init", type=int, default=0, help="initialization method type")
parser.add_argument("--bs", type=int, default=64, help="minibatch size")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--max_iter", type=int, default=200000, help="total iterations")
parser.add_argument("--neurons", type=int, default=50, help="neurons for MLP")
parser.add_argument("--num_workers", type=int, default=4, help="threads for loading dataset")
parser.add_argument("--overwrite", action="store_true", default=False, help="whether or not overwrite exist files")

args = parser.parse_args()


@torch.no_grad()
def initialize_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Initialize biases to zero


@torch.no_grad()
def initialize_xavier_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Initialize biases to zero


@torch.no_grad()
def initialize_kaiming_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Initialize biases to zero


@torch.no_grad()
def initialize_kaiming_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Initialize biases to zero


@torch.no_grad()
def initialize_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.5, 0.5)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Initialize biases to zero


@torch.no_grad()
def initialize_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 10)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Initialize biases to zero


def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    num_workers = args.num_workers
    # learning rate greater than 1.0 does not converge, 10x10 sample points will take about 320 hours
    init_type = args.init
    lr = args.lr
    bs = args.bs
    max_iter = args.max_iter
    neurons = args.neurons
    overwrite = args.overwrite
    test_point_indices = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
    save_file = f'./autodl-tmp/train_path_init{init_type}_lr{lr}_bs{bs}_neurons{neurons}_max_iter{max_iter}.npz'
    if os.path.exists(save_file) and not overwrite:
        print(f"file: {save_file} exits, exit......", flush=True)
        sys.exit(0)
    else:
        print(f"recording train_path_init{init_type}_lr{lr}_bs{bs}_neurons{neurons}_max_iter{max_iter}")

    train_dataset = torchvision.datasets.MNIST(
        root='./MNIST',
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./MNIST',
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        download=True
    )

    start_time = time.time()

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, neurons),
        nn.ReLU(),
        nn.Linear(neurons, neurons),
        nn.ReLU(),
        nn.Linear(neurons, 10)
    )

    initialization_methods = [
        initialize_xavier_uniform,
        initialize_xavier_normal,
        initialize_kaiming_uniform,
        initialize_kaiming_normal,
        initialize_uniform,
        initialize_normal,
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.train()
    save_file = f"mnist_model_init{init_type}_neurons{neurons}.pt"
    if os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file, weights_only=True))
    else:
        model.apply(initialization_methods[init_type])
        torch.save(model.state_dict(), save_file)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.data.data.shape[0], shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)

    optimizer = optim.SGD(model.parameters(), lr=lr)  # use SGD instead of Adam for better understanding of noise
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=0.0001,
    # )
    criterion = nn.CrossEntropyLoss()
    n_samples = len(train_loader)

    training_path = []
    losses = []
    accuracies = []

    epoch = 0
    flag = False
    num_iters = 0
    training_path.append(vectorize_weights(model))
    while not flag:
        epoch += 1
        loss = 0
        test_accuracy = 0

        for images, labels in train_loader:
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # already divided by the batch size
            loss.backward()
            optimizer.step()
            num_iters += 1
            losses.append(loss.item())
            if num_iters in test_point_indices:
                training_path.append(vectorize_weights(model))
                test_accuracy = calculate_accuracy(model, test_loader, device)
                accuracies.append(test_accuracy)
                print(f"iter {num_iters}, Epoch [{epoch}], Loss: {loss.item():.4f}, ACC: {test_accuracy:.4f}", flush=True)
            if num_iters >= test_point_indices[-1]:
                flag = True
                break

    training_path = np.stack(training_path).squeeze()
    save_file = f'./autodl-tmp/train_path_init{init_type}_lr{lr}_bs{bs}_neurons{neurons}_max_iter{max_iter}.npz'

    np.savez(save_file,
             training_path=training_path,
             losses=np.array(losses),
             accuracy=np.array(accuracies),
             record_iter_indices=test_point_indices,
             )

    end_time = time.time()
    hours = (end_time - start_time) / 3600
    print(f"Used time: {hours:.2f} hours")
