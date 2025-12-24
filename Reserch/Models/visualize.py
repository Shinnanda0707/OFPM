import argparse
import os
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import (
    resnet18,
    regnet_x_1_6gf,
    efficientnet_b2,
    mobilenet_v3_large,
)
from torch.utils.data import Dataset
import cv2

torch.cuda.empty_cache()


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
            self.image_dir_list[idx],
        ]

        return sample


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


torch.cuda.empty_cache()
print(torch.__version__)
print(torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="RegNet")
parser.add_argument("--epoch", "-e", type=str, default="70")
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--num_workers", "-nw", type=int, default=12)
parser.add_argument("--root", "-r", type=str, default="Dataset/Crash/Crash-1500/Frames")
parser.add_argument(
    "--anno", "-a", type=str, default="Dataset/Crash/Crash-1500/Crash-1500.txt",
)
args = parser.parse_args()

state_dict_file = ""

print("Running on ", end="")
if args.model == "CNN":
    print("CNN")
    net = CNN()
    state_dict_file = os.path.join("Models/checkpoint/CNN", f"{args.epoch}.pth")

elif args.model == "ResNet":
    print("ResNet18")
    net = resnet18(True)
    net.fc = nn.Linear(512, out_features=2)
    state_dict_file = os.path.join(
        "Models/checkpoint/ResNet/Final", f"{args.epoch}.pth"
    )

elif args.model == "RegNet":
    print("RegNetX 1.6gf")
    net = regnet_x_1_6gf(True)
    net.fc = nn.Linear(912, 2)
    state_dict_file = os.path.join("Models/checkpoint/RegNet", f"{args.epoch}.pth")

elif args.model == "EfficientNet":
    print("EfficientNet B2")
    net = efficientnet_b2(num_classes=2)
    state_dict_file = os.path.join(
        "Models/checkpoint/EfficientNet/Final", f"{args.epoch}.pth"
    )

elif args.model == "MobileNet":
    print("MobileNet v3 (large)")
    net = mobilenet_v3_large(num_classes=2)
    state_dict_file = os.path.join("Models/checkpoint/MobileNet", f"{args.epoch}.pth")

else:
    raise ValueError(f"{args.model} is NOT a valid model!")

net.load_state_dict(torch.load(state_dict_file))
net.cuda()
net.eval()

test_dataset = crash_dataset(
    anno_file=args.anno,
    root_dir=os.path.join(args.root, "Test"),
    transform=transforms.Compose(
        [transforms.Resize(size=(256, 512)), transforms.ToTensor(),]
    ),
    sub_sampling_freq=1,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

softmax = nn.Softmax(dim=1)

"""with open("Dataset/Label.txt", "a", encoding="UTF-8") as f:
    for i, data in enumerate(test_loader):
        print(f"\r{i}", end="")

        image, _, _ = data
        image = image.cuda()
        prediction = softmax(net(image))

        for pred in prediction.tolist():
            f.write(f"{round(pred[0], 4)}:{round(pred[1], 4)}\n")
    print()
f.close()"""

with open("Dataset/Label.txt", "r", encoding="UTF-8") as f:
    labels = str(f.read()).strip().split("\n")
    out = cv2.VideoWriter(
        "Dataset/Crash/Crash-1500/Frames/Visualization/prediction.avi",
        cv2.VideoWriter_fourcc(*"DIVX"),
        5,
        (1280, 720),
    )

    for i, data in enumerate(test_dataset):
        label = labels[i].split(":")

        img = cv2.imread(data[2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label[1], (60, 60), font, 4, (0, 0, 0), 10)

        out.write(img)

        print(f"\r{i}  {label[1]}", end="")
    out.release()
    print("\nDone!")

f.close()
