from enum import Enum

import cv2 as cv


WRIST_ID = 16
ELBOW_ID = 14
SHOULDER_ID = 12

OTHER_WRIST_ID = 15
OTHER_ELBOW_ID = 13
OTHER_SHOULDER_ID = 11

class Success(Enum):
    SUCCESS = 1
    FAIL = 2


class Direction(Enum):
    UP = 1
    DOWN = 2


class Color(Enum):
    # BGR
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


def draw_circle(img, cx, cy, color, radius=5, is_filled=True) -> None:
    if isinstance(cx, float):
        cx = int(cx)
    if isinstance(cy, float):
        cy = int(cy)

    assert isinstance(color, Color)
    if is_filled:
        cv.circle(img, (cx, cy), radius, color.value, cv.FILLED)
    else:
        cv.circle(img, (cx, cy), radius, color.value)


def show_image(img, title="", duration=0) -> None:
    cv.imshow(title, img)
    cv.waitKey(duration)
