import os
from random import random
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision
from torchvision import transforms

from torchvision.models import (
    resnet18,
    regnet_x_1_6gf,
    efficientnet_b2,
    mobilenet_v3_large,
)
from Models.ResNet_3D import BasicBlock, Bottleneck
from Models.ResNet_3D import resnet18 as rs18
from torch.utils.tensorboard import SummaryWriter


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


class crash_dataset_3D_optical_flow(Dataset):
    def __init__(self, root, anno_file, transform=None):
        self.root = root
        self.anno_file = anno_file
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, index):
        with open(self.anno_file, "r", encoding="UTF-8") as f:
            video = []

            for frame in range(1, 48, 2):
                """Dataset video's frame starts from index 1 
                beacause there can't be optical flow map for frame index 0."""
                try:
                    img = Image.open(
                        os.path.join(self.root, str(index), f"{frame}.jpg")
                    )
                    opt = Image.open(
                        os.path.join(
                            "Dataset/video_dataset_opticalflow",
                            str(index),
                            f"{frame - 1}.jpg",
                        )
                    )

                    if not (self.transform is None):
                        img = self.transform(img)
                        opt = self.transform(opt)
                    video.append(img.tolist())
                    video.append(opt.tolist())

                except FileNotFoundError as _:
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


class crash_dataset_3D(Dataset):
    def __init__(self, root, anno_file, transform=None):
        self.root = root
        self.anno_file = anno_file
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, index):
        with open(self.anno_file, "r", encoding="UTF-8") as f:
            video = []

            for frame in range(0, 49, 2):
                try:
                    img = Image.open(
                        os.path.join(self.root, str(index), f"{frame}.jpg")
                    )

                    if not (self.transform is None):
                        img = self.transform(img)
                    video.append(img.tolist())

                except FileNotFoundError as _:
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


class crash_dataset_2D(Dataset):
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


def acc(net, loader):
    correct, correct_vid, total = 0, 0, 0
    predicted_ls = []

    print("Start calculating accuracy of net")
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data

            images = images.cuda()
            outputs = net(images)

            labels = labels.cuda()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for p in predicted.tolist():
                predicted_ls.append(p)

        # print(len(predicted_ls))
        for video_index in range(0, len(predicted_ls), 50):
            for frame_index in range(50):
                if predicted_ls[video_index + frame_index]:
                    correct_vid += 1
                    break

    image_acc = 100 * correct / total
    video_acc = 50 * 100 * correct_vid / total

    return (image_acc, video_acc)


# State Dict
state_dict_root = {
    "2D_CNN": "Models/checkpoint/CNN",
    "2D_ResNet": "Models/checkpoint/ResNet/Final",
    "2D_RegNet": "Models/checkpoint/RegNet/L2_4e-1",
    "2D_EfficientNet": "Models/checkpoint/EfficientNet/Final",
    "2D_MobileNet": "Models/checkpoint/MobileNet",
}

# Models
cnn = CNN()
cnn.cuda()

resnet = resnet18()
resnet.fc = nn.Linear(512, 2)
resnet.cuda()

efficientnet = efficientnet_b2(num_classes=2)
efficientnet.cuda()

regnet = regnet_x_1_6gf()
regnet.fc = nn.Linear(912, 2)
regnet.cuda()

mobilenet = mobilenet_v3_large(num_classes=2)
mobilenet.cuda()

resnet.eval()
efficientnet.eval()
regnet.eval()
mobilenet.eval()

# Dataset
transform = [
    transforms.Resize(size=(256, 512)),
    transforms.ToTensor(),
]

train_dataset = crash_dataset_2D(
    anno_file="Dataset/Crash/Crash-1500/Crash-1500.txt",
    root_dir=os.path.join("Dataset/Crash/Crash-1500/Frames", "Train"),
    transform=transforms.Compose(transform),
    sub_sampling_freq=1,
)

val_dataset = crash_dataset_2D(
    anno_file="Dataset/Crash/Crash-1500/Crash-1500.txt",
    root_dir=os.path.join("Dataset/Crash/Crash-1500/Frames", "Validation"),
    transform=transforms.Compose(transform),
    sub_sampling_freq=2,
)

train_loader = DataLoader(train_dataset, 1, num_workers=12, drop_last=True)
val_loader = DataLoader(val_dataset, 1, num_workers=12, drop_last=True)

# Return
with torch.no_grad():
    """tb = SummaryWriter()
    for i in range(1, 71):
        cnn.load_state_dict(
            torch.load(os.path.join(state_dict_root["2D_CNN"], f"{i}.pth"))
        )
        cnn.eval()
        cnn_train_acc_img, cnn_train_acc_vid = acc(cnn, train_loader)
        cnn_val_acc_img, cnn_val_acc_vid = acc(cnn, val_loader)

        tb.add_scalar("Final Train Acc(Image)", cnn_train_acc_img, i)
        tb.add_scalar("Final Train Acc(Video)", cnn_train_acc_vid, i)
        tb.add_scalar("Fianl Val Acc(Image)", cnn_val_acc_img, i)
        tb.add_scalar("Fianl Val Acc(Video)", cnn_val_acc_vid, i)
    tb.close()

    tb = SummaryWriter()
    for i in range(1, 71):
        resnet.load_state_dict(
            torch.load(os.path.join(state_dict_root["2D_ResNet"], f"{i}.pth"))
        )
        resnet.eval()
        resnet_train_acc_img, resnet_train_acc_vid = acc(resnet, train_loader)
        resnet_val_acc_img, resnet_val_acc_vid = acc(resnet, val_loader)

        tb.add_scalar("Final Train Acc(Image)", resnet_train_acc_img, i)
        tb.add_scalar("Final Train Acc(Video)", resnet_train_acc_vid, i)
        tb.add_scalar("Fianl Val Acc(Image)", resnet_val_acc_img, i)
        tb.add_scalar("Fianl Val Acc(Video)", resnet_val_acc_vid, i)
    tb.close()"""

    tb = SummaryWriter()
    for i in range(1, 71):
        efficientnet.load_state_dict(
            torch.load(os.path.join(state_dict_root["2D_EfficientNet"], f"{i}.pth"))
        )
        efficientnet.eval()
        efficientnet_train_acc_img, efficientnet_train_acc_vid = acc(
            efficientnet, train_loader
        )
        efficientnet_val_acc_img, efficientnet_val_acc_vid = acc(
            efficientnet, val_loader
        )

        tb.add_scalar("Final Train Acc(Image)", efficientnet_train_acc_img, i)
        tb.add_scalar("Final Train Acc(Video)", efficientnet_train_acc_vid, i)
        tb.add_scalar("Fianl Val Acc(Image)", efficientnet_val_acc_img, i)
        tb.add_scalar("Fianl Val Acc(Video)", efficientnet_val_acc_vid, i)
    tb.close()

    tb = SummaryWriter()
    for i in range(1, 71):
        regnet.load_state_dict(
            torch.load(os.path.join(state_dict_root["2D_RegNet"], f"{i}.pth"))
        )
        regnet.eval()
        regnet_train_acc_img, regnet_train_acc_vid = acc(regnet, train_loader)
        regnet_val_acc_img, regnet_val_acc_vid = acc(regnet, val_loader)

        tb.add_scalar("Final Train Acc(Image)", regnet_train_acc_img, i)
        tb.add_scalar("Final Train Acc(Video)", regnet_train_acc_vid, i)
        tb.add_scalar("Fianl Val Acc(Image)", regnet_val_acc_img, i)
        tb.add_scalar("Fianl Val Acc(Video)", regnet_val_acc_vid, i)
    tb.close()

    tb = SummaryWriter()
    for i in range(1, 71):
        mobilenet.load_state_dict(
            torch.load(os.path.join(state_dict_root["2D_MobileNet"], f"{i}.pth"))
        )
        mobilenet.eval()
        mobilenet_train_acc_img, mobilenet_train_acc_vid = acc(mobilenet, train_loader)
        mobilenet_val_acc_img, mobilenet_val_acc_vid = acc(mobilenet, val_loader)

        tb.add_scalar("Final Train Acc(Image)", mobilenet_train_acc_img, i)
        tb.add_scalar("Final Train Acc(Video)", mobilenet_train_acc_vid, i)
        tb.add_scalar("Fianl Val Acc(Image)", mobilenet_val_acc_img, i)
        tb.add_scalar("Fianl Val Acc(Video)", mobilenet_val_acc_vid, i)
    tb.close()
