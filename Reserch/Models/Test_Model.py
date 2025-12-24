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
        if anno_file is not None:
            self.anno_list = read_anno_file(anno_file)
        else:
            self.anno_list = None
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
                self.image_dir_list.append(image_dir)

                if self.anno_list is not None:
                    label = self.anno_list[int(video_index)][frame_index]
                    self.label_list.append(label)

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, idx):
        if self.anno_list is not None:
            sample = [
                self.transform(Image.open(self.image_dir_list[idx])),
                self.label_list[idx],
                # self.image_dir_list[idx], # Uncomment when checking dataset labels (When calling: check_dataset())
            ]
        else:
            sample = self.transform(Image.open(self.image_dir_list[idx]))

        return sample


"""class crash_dataset_2D(Dataset):
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
            sample = [
                torch.FloatTensor(video),
                int((f.readlines()[index]).strip().replace("\n", "").split(":")[1]),
            ]
        f.close()
        del video

        return sample"""


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


def test_3D(net, data_loader):
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

    try:
        precision = true_positive / (true_positive + false_positive)
    except:
        precision = -1

    try:
        recall = true_positive / (true_positive + false_negative)
    except:
        recall = -1

    return (correct, precision, recall)


"""def test_2D(net, data_loader):
    total = 0
    correct = 0
    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        inputs = inputs.cuda()
        out = 0
        for dk in inputs:
            for d in dk:
                outputs = net(d.reshape([1, 3, 256, 512]))
                _, predicted = torch.max(outputs.data, 1)
                if predicted == 1:
                    out = predicted

        labels = labels.cuda()

        total += labels.size(0)
        correct += (out == labels).sum().item()

        labels = labels.tolist()[0]

        if out:
            if out == labels:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if out == labels:
                true_negative += 1
            else:
                false_negative += 1

    print(
        f"tp: {true_positive}  fp: {false_positive}  tn: {true_negative}  fn: {false_negative}",
        end="",
    )

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return (correct, precision, recall)"""


def test_2D(net, data_loader):
    correct, correct_vid, total = 0, 0, 0
    predicted_ls = []

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data

            images = images.cuda()
            outputs = net(images)

            labels = labels.cuda()
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

        test_video_acc = 50 * 100 * correct_vid / total
        test_img_acc = 100 * correct / total
        # print()

    return (test_video_acc, test_img_acc)


def test_normal_condition_2D(net, data_loader):
    correct_vid, total = 0, 0
    predicted_ls = []

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images = data

            images = images.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            for p in predicted.tolist():
                predicted_ls.append(p)

        for video_index in range(0, len(predicted_ls), 50):
            normal = True
            for frame_index in range(50):
                if predicted_ls[video_index + frame_index]:
                    normal = False
                    break
            if normal:
                correct_vid += 1

        test_video_acc = 100 * correct_vid / 500
        # print()

    return test_video_acc


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

resnet_3 = rs18(False, False, False, None, BasicBlock)
resnet_3.cuda()
ofpm_resnet_3 = rs18(
    False, False, True, nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Bottleneck
)
ofpm_resnet_3.cuda()

cnn.load_state_dict(torch.load("Models/checkpoint/CNN/70.pth"))
resnet.load_state_dict(torch.load("Models/checkpoint/ResNet/Final/70.pth"))
efficientnet.load_state_dict(torch.load("Models/checkpoint/EfficientNet/Final/70.pth"))
regnet.load_state_dict(torch.load("Models/checkpoint/RegNet/L2_4e-1/70.pth"))
mobilenet.load_state_dict(torch.load("Models/checkpoint/MobileNet/70.pth"))

resnet_3.load_state_dict(torch.load("Models/checkpoint/3D_ResNet/Final/70.pth"))
ofpm_resnet_3.load_state_dict(torch.load("Models/checkpoint/With_optical_flow/70.pth"))

cnn.eval()
resnet.eval()
efficientnet.eval()
regnet.eval()
mobilenet.eval()

resnet_3.eval()
ofpm_resnet_3.eval()

transform = [
    transforms.Resize(size=(256, 512)),
    transforms.ToTensor(),
]
dataset_2 = crash_dataset_2D(
    anno_file="Dataset/Crash/Crash-1500/Crash-1500.txt",
    root_dir=os.path.join("Dataset/Crash/Crash-1500/Frames", "Train"),
    transform=transforms.Compose(transform),
    sub_sampling_freq=1,
)
normal_dataset_2 = crash_dataset_2D(
    anno_file=None,
    root_dir=os.path.join("Dataset/Crash/Crash-1500/Frames", "Normal"),
    transform=transforms.Compose(transform),
    sub_sampling_freq=1,
)
"""dataset_2 = crash_dataset_2D(
    "Dataset/video_dataset", "Dataset/video_label.txt", transforms.Compose(transform)
)"""
"""dataset_3 = crash_dataset_3D(
    "Dataset/video_dataset", "Dataset/video_label.txt", transforms.Compose(transform)
)"""
dataset_3 = crash_dataset_3D(
    "Dataset/Crash/Crash-1500/Frames/Normal",
    "Dataset/Crash/Crash-1500/Frames/video_label.txt",
    transforms.Compose(transform),
)
dataset_3_opt = crash_dataset_3D_optical_flow(
    "Dataset/video_dataset", "Dataset/video_label.txt", transforms.Compose(transform)
)

print(len(dataset_3))
print(len(dataset_3_opt))

# _, dataset_2 = random_split(dataset_2, [2192 + 730 + 720, 10])
"""_, _, dataset_3 = random_split(
    dataset_3, [2192, 730, 730], generator=torch.Generator().manual_seed(42)
)"""
# _, dataset_3_opt = random_split(dataset_3_opt, [2192 + 730 + 720, 10])

data_loader_2 = DataLoader(dataset_2, batch_size=1)
normal_data_loader_2 = DataLoader(normal_dataset_2, batch_size=1)
data_loader_3 = DataLoader(dataset_3, batch_size=1)
data_loader_3_opt = DataLoader(dataset_3_opt, batch_size=1)

with torch.no_grad():
    """cnn_acc = test_normal_condition_2D(cnn, normal_data_loader_2)
    print(f"\nCNN: {cnn_acc}")
    resnet_acc = test_normal_condition_2D(resnet, normal_data_loader_2)
    print(f"\nResNet: {resnet_acc}")
    efficientnet_acc = test_normal_condition_2D(efficientnet, normal_data_loader_2)
    print(f"\nEfficientNet: {efficientnet_acc}")
    regnet_acc = test_normal_condition_2D(regnet, normal_data_loader_2)
    print(f"\nRegNet: {regnet_acc}")
    mobilenet_acc = test_normal_condition_2D(mobilenet, normal_data_loader_2)
    print(f"\nMobileNet: {mobilenet_acc}")"""

    resnet_3_acc = test_3D(resnet_3, data_loader_3)
    print(f"\n3D ResNet: {resnet_3_acc}")
    """ofpm_resnet_3_acc = test_3D(ofpm_resnet_3, data_loader_3_opt)
    print(f"\nOFPM 3D ResNet: {ofpm_resnet_3_acc}")"""
