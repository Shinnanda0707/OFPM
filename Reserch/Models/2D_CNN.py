import os
import time
import argparse
from PIL import Image

import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet18,
    regnet_x_1_6gf,
    efficientnet_b2,
    mobilenet_v3_large,
)
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()
print(torch.__version__)
print(torchvision.__version__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.max_pool = nn.MaxPool2d(5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5)

        self.fc1 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 2)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.max_pool(self.activation(self.conv1(x)))
        x = self.max_pool(self.activation(self.conv2(x)))
        x = self.max_pool(self.activation(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x


def read_anno_file(anno_file):
    assert os.path.exists(anno_file), "Annotation file does not exist!" + anno_file

    result_ls = []
    total_accident = 0
    with open(anno_file, "r") as f:
        for line in f.readlines():
            labels = line.strip().split(",[")[1].split("],")[0].split(",")
            for i in range(len(labels)):
                labels[i] = int(labels[i].strip())
                total_accident += labels[i]
            result_ls.append(labels)
    f.close()

    print(total_accident)

    return result_ls


class crash_dataset(Dataset):
    def __init__(self, anno_file, root_dir, sub_sampling_freq, transform=None):
        self.anno_list = read_anno_file(anno_file)
        self.root_dir = root_dir
        self.sub_sampling_freq = sub_sampling_freq
        self.transform = transform
        self.image_dir_list = []
        self.label_list = []

        for video_index in os.listdir(self.root_dir):
            parent_dir = os.path.join(self.root_dir, video_index)
            for frame_index in range(
                0, len(os.listdir(parent_dir)), self.sub_sampling_freq
            ):
                image_dir = os.path.join(parent_dir, f"{frame_index}.jpg")
                label = self.anno_list[int(video_index)][frame_index]

                self.image_dir_list.append(image_dir)
                self.label_list.append(label)

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, idx):
        sample = [
            self.transform(Image.open(self.image_dir_list[idx])),
            self.label_list[idx],
            # self.image_dir_list[idx], # Uncomment when checking dataset labels (When calling: check_dataset())
        ]

        return sample


def train(net, crit, opt, train_loader):
    correct, total, train_loss = 0, 0, 0.0

    print("Start training net")
    net.train()
    for i, data in enumerate(train_loader):
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

        # print(f"\r{i + 1} / {len(train_loader)}   Current: {pred.tolist()}", end="")

    train_loss /= len(train_loader)
    train_acc = 100 * correct / total
    print()

    return (train_loss, train_acc)


def val(net, crit, val_loader):
    correct, correct_vid, total, val_loss = 0, 0, 0, 0.0
    predicted_ls = []

    print("Start validating net")
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            images, labels = data

            images = images.cuda()
            outputs = net(images)

            labels = labels.cuda()
            loss = crit(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for p in predicted.tolist():
                predicted_ls.append(p)

            # print(f"\r{i} / {len(val_loader)}   Current: {predicted.tolist()}", end="")

        for video_index in range(0, len(predicted_ls), 50):
            for frame_index in range(50):
                if predicted_ls[video_index + frame_index]:
                    correct_vid += 1
                    break

        val_image_loss = val_loss / len(val_loader)
        val_image_acc = 100 * correct / total
        val_video_acc = 50 * 100 * correct_vid / total
        print()

    return (val_image_loss, val_image_acc, val_video_acc)


def check_dataset(dataset):
    images = []
    for i, data in enumerate(dataset):
        print(f"\r{i}  {data[1]}", end="")
        img = cv2.imread(data[2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(data[1]), (20, 20), font, 4, (0, 0, 0), 10)
        images.append(img)

    w, h, _ = images[0].shape
    size = (h, w)
    out = cv2.VideoWriter(
        "Dataset/Crash/Crash-1500/Frames/Visualization/vid.avi",
        cv2.VideoWriter_fourcc(*"DIVX"),
        1,
        size,
    )

    for img in images:
        out.write(img)
    out.release()
    print("\nDone!")


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="EfficientNet")
parser.add_argument("--opt", "-op", type=str, default="RMSprop")
parser.add_argument(
    "--anno", "-a", type=str, default="Dataset/Crash/Crash-1500/Crash-1500.txt",
)
parser.add_argument("--root", "-r", type=str, default="Dataset/Crash/Crash-1500/Frames")
parser.add_argument(
    "--checkpoint", "-cp", type=str, default="Models/checkpoint/EfficientNet"
)
parser.add_argument("--epoch", "-e", type=int, default=70)
parser.add_argument("--lr", "-lr", type=float, default=1e-4)
parser.add_argument("--batch_size", "-bs", type=int, default=64)
parser.add_argument("--num_workers", "-nw", type=int, default=12)
parser.add_argument("--weight_decay", "-l2", type=float, default=0.0)
parser.add_argument("--SGD_momentum", "-sm", type=float, default=0.8)
parser.add_argument("--sub_sampling_freq", "-sf", type=int, default=1)
args = parser.parse_args()

# Dataset
transform = transforms.Compose(
    [transforms.Resize(size=(256, 512)), transforms.ToTensor(),]
)

train_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAutocontrast(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ]
)

train_dataset = crash_dataset(
    anno_file=args.anno,
    root_dir=os.path.join(args.root, "Train"),
    transform=train_transform,
    sub_sampling_freq=args.sub_sampling_freq,
)

val_dataset = crash_dataset(
    anno_file=args.anno,
    root_dir=os.path.join(args.root, "Validation"),
    transform=transform,
    sub_sampling_freq=1,
)

test_dataset = crash_dataset(
    anno_file=args.anno,
    root_dir=os.path.join(args.root, "Test"),
    transform=transform,
    sub_sampling_freq=1,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

print(len(train_dataset), len(test_dataset), len(val_dataset))
# check_dataset(val_dataset)

# Define model
print("Running on ", end="")
if args.model == "CNN":
    print("CNN")
    net = CNN()

elif args.model == "ResNet":
    print("ResNet18")
    net = resnet18(True)
    net.fc = nn.Linear(512, out_features=2)

elif args.model == "RegNet":
    print("RegNetX 1.6gf")
    net = regnet_x_1_6gf(True)
    net.fc = nn.Linear(912, 2)

elif args.model == "EfficientNet":
    print("EfficientNet B2")
    net = efficientnet_b2(num_classes=2)

elif args.model == "MobileNet":
    print("MobileNet v3 (large)")
    net = mobilenet_v3_large(num_classes=2)

else:
    raise ValueError(f"{args.model} is NOT a valid model!")
net.cuda()

if args.opt == "RMSprop":
    opt = optim.RMSprop(
        params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
elif args.opt == "Adam":
    opt = optim.Adam(
        params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
elif args.opt == "SGD":
    opt = optim.SGD(
        params=net.parameters(),
        lr=args.lr,
        momentum=args.SGD_momentum,
        weight_decay=args.weight_decay,
    )
else:
    raise ValueError(f"{args.opt} is NOT a valid optimizer!")

crit = nn.CrossEntropyLoss()

# Tensorboard
tb = SummaryWriter()

# Train
for epoch in range(args.epoch):
    ts = time.time()

    # Train Model
    train_loss, train_acc = train(net, crit, opt, train_loader)

    # Validate Model
    val_image_loss, val_image_acc, val_video_acc = val(net, crit, val_loader)

    # Save Model
    torch.save(net.state_dict(), os.path.join(args.checkpoint, f"{epoch + 1}.pth"))

    te = time.time()

    # Add elements to tb
    tb.add_scalar("Train Acc", train_acc, epoch)
    tb.add_scalar("Train Loss", train_loss, epoch)
    tb.add_scalar("Val Acc (Image)", val_image_acc, epoch)
    tb.add_scalar("Val Acc (Video)", val_video_acc, epoch)
    tb.add_scalar("Val Loss (Image)", val_image_loss, epoch)

    print("Done!")

    # Log status
    print(f"\n{'=' * 50}[Summary]{'=' * 50}")
    print(f"Epoch:                 {epoch + 1} / {args.epoch}")
    print(f"Train Acc:             {train_acc}")
    print(f"Val Acc (img / vid):   {val_image_acc} / {val_video_acc}")
    print(f"Loss    (train / val): {train_loss} / {val_image_loss}")
    print(f"Took {int(te - ts)} seconds")
    print("=" * 109)

# Close tb
tb.close()

# Delete model
with torch.no_grad():
    net.cpu()
    del net
    torch.cuda.empty_cache()
del crit
del opt
