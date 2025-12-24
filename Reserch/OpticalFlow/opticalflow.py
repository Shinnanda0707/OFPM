import os
import numpy as np

import cv2 as cv

"""video_idx = 0
for video in os.listdir("Dataset/Crash/Crash-1500/Crash"):
    print(f"\r{video}", end="")

    cap = cv.VideoCapture(
        cv.samples.findFile(f"Dataset/Crash/Crash-1500/Crash/{video}")
    )
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame = 0

    os.mkdir(f"Dataset/video_dataset_opticalflow/{video_idx}")
    while 1:
        ret, frame2 = cap.read()

        if not ret:
            print("Done!")
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imwrite(f"Dataset/video_dataset_opticalflow/{video_idx}/{frame}.jpg", bgr)
        prvs = next
        frame += 1

    cv.destroyAllWindows()
    video_idx += 1"""

video_idx = 1500
for video in os.listdir("Dataset/Crash/Crash-1500/Normal"):
    print(f"\r{video}", end="")

    cap = cv.VideoCapture(
        cv.samples.findFile(f"Dataset/Crash/Crash-1500/Normal/{video}")
    )
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame = 0

    os.mkdir(f"Dataset/video_dataset_opticalflow/{video_idx}")
    while 1:
        ret, frame2 = cap.read()

        if not ret:
            print("Done!")
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imwrite(f"Dataset/video_dataset_opticalflow/{video_idx}/{frame}.jpg", bgr)
        prvs = next
        frame += 1

    cv.destroyAllWindows()
    video_idx += 1
