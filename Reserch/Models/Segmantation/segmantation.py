import os
import numpy as np
from PIL import Image

import cv2
import torch
from torchvision.models.segmentation import fcn_resnet101
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# Define the helper function
def decode_segmap(image, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (0, 0, 0),
            (128, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 0),
            (0, 0, 128),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (128, 128, 0),
            (128, 0, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ]
    )
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, path, save_loc):
    img = Image.open(path)

    transform = Compose(
        [
            Resize(size=(256, 512)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inp = transform(img).unsqueeze(0)
    out = net(inp)["out"]
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    image = Image.fromarray(rgb)
    image.save(save_loc)


# seg = deeplabv3_resnet101(True, True).eval()
seg = fcn_resnet101(True, True).eval()

for vid in os.listdir("Dataset/video_dataset"):
    print(f"\rProcessing {vid}", end="")
    os.mkdir(f"Dataset/video_dataset_segmanted/{vid}")
    for img in os.listdir(f"Dataset/video_dataset/{vid}"):
        segment(
            seg,
            f"Dataset/video_dataset/{vid}/{img}",
            f"Dataset/video_dataset_segmanted/{vid}/{img}",
        )
