import math

import mediapipe as mp
import cv2 as cv


class PoseDetector():

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.results = None
        
        static_image_mode = False       # This is a video
        model_complexity = 2                # 0, 1, or 2 (2 is the most accurate but also the slowest)
        smooth_landmarks = True          # Smooth landmarks to reduce jitter (for videos only)
        enable_segmentation = False
        smooth_segmentation = False

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode, model_complexity, smooth_landmarks, 
            enable_segmentation, smooth_segmentation, min_detection_confidence, min_tracking_confidence)
        

    def find_pose(self, img, draw=False, ids_to_draw=None):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def find_position(self, img, draw=False):
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=False):
        # Get the landmarks
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return angle
