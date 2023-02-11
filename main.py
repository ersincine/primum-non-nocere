from enum import Enum

import numpy as np
import cv2 as cv

import pose_module as pm
from utils import *


VIDEO_PATH = "video.mp4"
VIDEO_START_SEC = 5
VIDEO_DURATION_SEC = 5

# Resolution of the screen maybe. Not very important.
MAX_WIDTH = 1024
MAX_HEIGHT = 768

ANGLE_CHECK_DURATION_SEC = 3
ANGLE_TOLERANCE_DEG = 5

MIN_DETECTION_CONFIDENCE = 0.9
MIN_TRACKING_CONFIDENCE = 0.9


capture = cv.VideoCapture(VIDEO_PATH)
if capture is None or not capture.isOpened():
    print("Error opening video file.")
    exit(1)

framerate = round(capture.get(cv.CAP_PROP_FPS))

print("Framerate:", framerate, "FPS")

arm_ids = [SHOULDER_ID, ELBOW_ID, WRIST_ID]
direction_reference_ids = [SHOULDER_ID, ELBOW_ID, WRIST_ID]  # TODO: Consider changing this. Maybe fingers, face, etc. (Or just one of these.)

detector = pm.PoseDetector(min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE)


width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
print("Original resolution:", width, "x", height)
if width > MAX_WIDTH or height > MAX_HEIGHT:
    if width / MAX_WIDTH > height / MAX_HEIGHT:
        height = int(MAX_WIDTH * height / width)
        width = MAX_WIDTH
    else:
        width = int(MAX_HEIGHT * width / height)
        height = MAX_HEIGHT
print("New resolution:", width, "x", height)

angles = []

capture.set(cv.CAP_PROP_POS_FRAMES, VIDEO_START_SEC * framerate)

prev_mean_y = None
prev_direction = None

num_pressures = 0
angle_success = True

for _ in range(VIDEO_DURATION_SEC * framerate):

    suc, img = capture.read()
    img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

    detector.find_pose(img, draw=False)
    landmarks = detector.find_position(img, draw=False)

    if len(landmarks) != 0:
        angle = detector.find_angle(img, *arm_ids, draw=True)
        angles.append(angle)

        if len(angles) >= ANGLE_CHECK_DURATION_SEC * framerate:  # FIXME: We skip some frames, so it more than ANGLE_CHECK_DURATION_SEC seconds.
            mean_angle = np.mean(angles[-ANGLE_CHECK_DURATION_SEC * framerate:])
            if abs(180-mean_angle) > ANGLE_TOLERANCE_DEG:
                angle_success = False

        all_x = []
        all_y = []
        for id, cx, cy in landmarks:
            if id in direction_reference_ids:
                all_x.append(cx)
                all_y.append(cy)
                draw_circle(img, cx, cy, Color.BLUE)

        mean_x = np.mean(all_x)
        mean_y = np.mean(all_y)

        if prev_mean_y is not None:
            if mean_y > prev_mean_y:
                current_direction = Direction.DOWN
            else:
                current_direction = Direction.UP
            print(current_direction)
            if prev_direction is not None:
                if prev_direction == Direction.DOWN and current_direction == Direction.UP:
                    # TODO: Three consecutive DOWNs should be considered DOWN. Then, three consecutive UPs should be considered UP. Only then we can count a pressure.
                    num_pressures += 1
            prev_direction = current_direction

        draw_circle(img, mean_x, mean_y, Color.GREEN)
        prev_mean_y = mean_y

    show_image(img, duration=1)

print("mean of angles:", np.mean(angles))
print("standard deviation of angles:", np.std(angles))
print(f"number of pressures: {num_pressures} ({round((num_pressures * 60) / VIDEO_DURATION_SEC)} per minute)")
# TODO: Calculate the number of pressures per minute not once but for every 10 seconds or so.
print("angle success:", angle_success)
