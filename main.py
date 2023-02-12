from enum import Enum

import numpy as np
import cv2 as cv

import pose_module as pm
from utils import *


# Select and crop the video.
VIDEO_PATH = "video.mp4"
VIDEO_START_SEC = 5
VIDEO_DURATION_SEC = 5

# ANGLES
# Some arbitrary values below. 
# Let's call the angle between the shoulder, elbow and wrist just "angle".
# Throughout the video, the average of angles over the last ANGLE_CHECK_DURATION_SEC seconds should 
# always be greater than or equal to 180-ANGLE_TOLERANCE_DEG.
ANGLE_CHECK_DURATION_SEC = 3
ANGLE_TOLERANCE_DEG = 5
NUM_FRAMES_FOR_ANGLE_SMOOTHING = 3  # This is not for evaluation, but for obtaining more accurate angles. (Her an son üç karenin ortalaması gösterilecek.)

# COMPRESSIONS
NUM_FRAMES_FOR_COORDINATE_SMOOTHING = 3  # This is not for evaluation, but for obtaining more accurate directions. (Her an son üç karenin ortalaması kullanılacak.)
NUM_CONSECUTIVE_DIRECTIONS_TO_EXPECT = 3  # This many consecutive directions are expected to be the same.

# Higher values mean more accurate pose estimation but fewer frames with a pose are detected. (I think.)
MIN_DETECTION_CONFIDENCE = 0.9
MIN_TRACKING_CONFIDENCE = 0.9

# Resolution of the screen maybe. Not very important.
MAX_WIDTH = 1024
MAX_HEIGHT = 768


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

capture.set(cv.CAP_PROP_POS_FRAMES, VIDEO_START_SEC * framerate)

angles = []
stable_angles = []
angle_successes = []
# TODO: Angles are already stable thanks to the deterministic formula! Calculate the stable positions instead. Calculate the angles from the stable positions.

ys = []  # These are the spatially smoothed y coordinates of the direction_reference_ids.
stable_ys = []  # These are the temporally smoothed values of ys.
directions = []
stable_directions = []
num_compressions = 0

for _ in range(VIDEO_DURATION_SEC * framerate):

    success, img = capture.read()
    if not success:
        print("Video is shorter than expected.")
        exit(1)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

    detector.find_pose(img, draw=False)
    landmarks = detector.find_position(img, draw=False)

    if len(landmarks) != 0:
        angle = detector.find_angle(img, *arm_ids, draw=True)
        if angle > 180:
            angle = 360 - angle
        angles.append(angle)

        stable_angle = np.mean(angles[-NUM_FRAMES_FOR_ANGLE_SMOOTHING:])
        stable_angles.append(stable_angle)

        if len(stable_angles) >= ANGLE_CHECK_DURATION_SEC * framerate:  # FIXME: We skip some frames, so it more than ANGLE_CHECK_DURATION_SEC seconds.
            average_angle = np.mean(angles[-ANGLE_CHECK_DURATION_SEC * framerate:])
            if average_angle >= 180 - ANGLE_TOLERANCE_DEG:
                angle_successes.append(Success.SUCCESS)
            else:
                angle_successes.append(Success.FAILURE)

        all_x = []
        all_y = []
        for id, cx, cy in landmarks:
            if id in direction_reference_ids:
                all_x.append(cx)
                all_y.append(cy)
                draw_circle(img, cx, cy, Color.BLUE)

        x = np.mean(all_x)
        y = np.mean(all_y)
        draw_circle(img, x, y, Color.GREEN)
        ys.append(y)

        if len(ys) > NUM_FRAMES_FOR_COORDINATE_SMOOTHING:
            stable_ys.append(np.mean(ys[-NUM_FRAMES_FOR_COORDINATE_SMOOTHING:]))
            if len(stable_ys) >= 2:
                if stable_ys[-2] < stable_ys[-1]:
                    curr_direction = Direction.DOWN
                else:
                    curr_direction = Direction.UP
                directions.append(curr_direction)

                if len(directions) > NUM_CONSECUTIVE_DIRECTIONS_TO_EXPECT:
                    if len(set(directions[-NUM_CONSECUTIVE_DIRECTIONS_TO_EXPECT:])) == 1:
                        stable_directions.append(curr_direction)

                        if len(stable_directions) >= 2:
                            if stable_directions[-2] == Direction.DOWN and stable_directions[-1] == Direction.UP:
                                num_compressions += 1
        
    show_image(img, duration=1)

print(f"number of compressions: {num_compressions} ({round((num_compressions * 60) / VIDEO_DURATION_SEC, 1)} per minute)")
# TODO: Calculate the number of compressions per minute not once but for every 10 seconds or so.
print(f"angle success rate: {round(len([x for x in angle_successes if x == Success.SUCCESS]) / len(angle_successes) * 100, 1)}%")
# TODO: Calculate the depth of the compression as well. (Instant!)
