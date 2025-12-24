import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision
import pytorch_model_summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from torch.utils.tensorboard import SummaryWriter

from Models.ResNet_3D import BasicBlock, Bottleneck, resnet18
from Models.RegNet import regnet_y_400mf

# from ResNet_2plus1D import resnet18_2_1d
torch.cuda.empty_cache()
print(torch.__version__)
print(torchvision.__version__)


class crash_dataset(Dataset):
    def __init__(self, root, anno_file, transform=None):
        self.root = root
        self.anno_file = anno_file
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, index):
        with open(self.anno_file, "r", encoding="UTF-8") as f:
            video = []

            for frame in range(1, 48, args.sub_sampling_freq,):
                """Dataset video's frame starts from index 1 
                beacause there can't be optical flow map for frame index 0."""
                try:
                    img = Image.open(
                        os.path.join(self.root, str(index), f"{frame}.jpg")
                    )
                    """opt = Image.open(
                        os.path.join(
                            "Dataset/video_dataset_opticalflow",
                            str(index),
                            f"{frame - 1}.jpg",
                        )
                    )"""

                    if not (transform is None):
                        img = transform(img)
                        # opt = transform(opt)
                    video.append(img.tolist())
                    # video.append(opt.tolist())
                except FileNotFoundError as e:
                    break

            video = np.array(video)
            i, c, h, w = video.shape
            sample = [
                torch.FloatTensor(np.array(video).reshape(c, i, h, w)),
                int((f.readlines()[index]).strip().replace("\n", "").split(":")[1]),
            ]
        f.close()
        del video

        return sample


# Return Net
def return_model(args):
    if args.model == "ResNet18":
        return resnet18(block_type="Bottleneck")

    if args.model == "ResNet(2+1)":
        return r2plus1d_18(False, False)

    raise ValueError(f"Invalid model name: {args.model}")


# Return optimizer
def return_optim(args, net):
    if args.opt == "RMSprop":
        return optim.RMSprop(
            params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.opt == "SGD":
        return optim.SGD(
            params=net.parameters(),
            lr=args.lr,
            momentum=args.SGD_momentum,
            weight_decay=args.weight_decay,
        )

    if args.opt == "Adam":
        return optim.Adam(
            params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    raise ValueError(f"Invalid optimizer name: {args.opt}")


def train(net, crit, opt, train_loader):
    correct, total, train_loss = 0, 0, 0.0

    print("Start training net")
    net.train()
    for _, data in enumerate(train_loader):
        opt.zero_grad()

        inputs, labels = data
        inputs = inputs.cuda()
        outputs = net(inputs)

        labels = labels.cuda()
        loss = crit(outputs, labels)

        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

        # print(pred)

    train_loss /= len(train_loader)
    train_acc = 100 * correct / total
    print()

    return (train_loss, train_acc)


def val(net, crit, val_loader):
    correct, total, val_loss = 0, 0, 0.0

    print("Start validating net")
    net.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            inputs, labels = data

            inputs = inputs.cuda()
            outputs = net(inputs)

            labels = labels.cuda()
            loss = crit(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print(predicted)

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        print()

    return (val_loss, val_acc)


def test(net, data_loader):
    total = 0
    correct = 0
    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        inputs = torch.FloatTensor(inputs).cuda()
        outputs = net(inputs)

        labels = labels.cuda()
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_positive += (predicted == labels == 1).sum().item()
        false_positive += ((predicted != labels) and predicted).sum().item()
        false_negative += ((predicted != labels) and labels).sum().item()
        true_negative += (predicted == labels == 0).sum().item()

    print(
        f"tp: {true_positive}  fp: {false_positive}  tn: {true_negative}  fn: {false_negative}",
        end="",
    )

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return (correct, precision, recall)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="ResNet18")
parser.add_argument("--opt", "-op", type=str, default="RMSprop")
parser.add_argument(
    "--anno", "-a", type=str, default="Dataset/video_label.txt",
)
parser.add_argument("--root", "-r", type=str, default="Dataset/video_dataset")
parser.add_argument(
    "--checkpoint", "-cp", type=str, default="Models/checkpoint/3D_ResNet/Final",
)
parser.add_argument("--epoch", "-e", type=int, default=70)
parser.add_argument("--lr", "-lr", type=float, default=1e-4)
parser.add_argument("--batch_size", "-bs", type=int, default=2)
parser.add_argument("--num_workers", "-nw", type=int, default=12)
parser.add_argument("--weight_decay", "-l2", type=float, default=1e-4)
parser.add_argument("--SGD_momentum", "-sm", type=float, default=0.9)
parser.add_argument("--sub_sampling_freq", "-sf", type=int, default=2)
args = parser.parse_args()

# Make dataset
transform = transforms.Compose(
    [transforms.Resize(size=(256, 512)), transforms.ToTensor()]
)

dataset = crash_dataset(args.root, args.anno, transform,)

print(len(dataset))

train_set, val_set, test_set = random_split(
    dataset, [2192, 730, 730], generator=torch.Generator().manual_seed(42)
)

# Load data
train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)

val_loader = DataLoader(
    val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

test_loader = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

# Declare net, opt
net = resnet18(False, False, False, None, BasicBlock)
# net = regnet_y_400mf()
net.cuda()
print(torch.cuda.memory_summary())
opt = return_optim(args, net)
crit = nn.CrossEntropyLoss()

# Tensorboard
tb = SummaryWriter()

# Train
for epoch in range(args.epoch):
    ts = time.time()

    # Train Model
    train_loss, train_acc = train(net, crit, opt, train_loader)

    # Validate Model
    val_loss, val_acc = val(net, crit, val_loader)

    # Save Model
    torch.save(net.state_dict(), os.path.join(args.checkpoint, f"{epoch + 1}.pth"))

    te = time.time()

    # Add elements to tb
    tb.add_scalar("Train Acc", train_acc, epoch)
    tb.add_scalar("Train Loss", train_loss, epoch)
    tb.add_scalar("Val Loss", val_loss, epoch)
    tb.add_scalar("Val Acc", val_acc, epoch)

    print("Done!")

    # Log status
    print(f"\n{'=' * 50}[Summary]{'=' * 50}")
    print(f"Epoch:                 {epoch + 1} / {args.epoch}")
    print(f"Train Acc:             {train_acc}")
    print(f"Val Acc:   {val_acc}")
    print(f"Loss (train / val): {train_loss} / {val_loss}")
    print(f"Took {int(te - ts)} seconds")
    print("=" * 109)

# Teset Model
test_result = test(net, test_loader)
print(test_result)

# Close tb
tb.close()

# Delete model
with torch.no_grad():
    net.cpu()
    del net
    torch.cuda.empty_cache()
del crit
del opt
