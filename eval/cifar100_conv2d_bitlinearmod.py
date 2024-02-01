# Originated from: https://github.com/pytorch/examples/blob/main/mnist/main.py
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

script_dir = os.path.dirname(os.path.abspath(__file__))
lpm_dir = os.path.dirname(script_dir)
sys.path.append(lpm_dir)

from lpm.layers import BitLinearMod

# Various params.
model_name = "cifar100_conv2d_bitlinearmod"


# Experimental model:
#   Convolutions: torch.nn.Conv2d
#   Linear: BitLinearMod
# Run:
#   python eval/cifar100_conv2d_bitlinearmod.py --seed=1 --batch-size=64 --epochs=10 --lr=1.0e-3 --wd=1.0e-5
# Result:
#   Test set: Average loss: 3.1790, Accuracy: 2354/10000 (24%)
class ExperimentalModel(nn.Module):
    def __init__(
        self,
    ):
        super(ExperimentalModel, self).__init__()
        self.conv_a = nn.Conv2d(3, 16, 3, 1, 1)
        self.pooling_a = nn.MaxPool2d(2, 2)
        self.conv_b = nn.Conv2d(16, 32, 3, 1, 1)
        self.pooling_b = nn.MaxPool2d(2, 2)
        self.conv_reductor = nn.Conv2d(32, 4, 1, 1, 0)
        self.relu = nn.ReLU()
        self.fc1 = BitLinearMod(
            in_features=256,
            out_features=84,
            fp8_e4m3=True,
        )
        self.fc2 = BitLinearMod(
            in_features=84,
            out_features=100,
            fp8_e4m3=True,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.conv_a(x)
        x = self.relu(x)
        x = self.pooling_a(x)
        x = self.conv_b(x)
        x = self.relu(x)
        x = self.pooling_b(x)
        x = self.conv_reductor(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="SelfProjection CIFAR-100 evaluation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        metavar="LR",
        help="learning rate (default: 1.0e-3)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1.0e-5,
        metavar="WD",
        help="Weight decay (default: 1.0e-5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.CIFAR100(
        f"{lpm_dir}/data", train=True, download=True, transform=transform
    )
    dataset_eval = datasets.CIFAR100(
        f"{lpm_dir}/data", train=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_eval, **test_kwargs)

    model = ExperimentalModel().to(device)

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total number of trainable parameters: {total_trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), f"model_{model_name}.pt")


if __name__ == "__main__":
    main()
