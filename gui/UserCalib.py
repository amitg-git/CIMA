import cv2
import numpy as np
import mediapipe as mp
import time

from cima.cima import CIMA
from helper.helper import *
from helper.monitor import monitor


class UserCalib:
    def __init__(self, cima: CIMA):
        self.frame = None
        self.radius = 25
        self.radius_focus = 8
        self.bbox_size = 100  # Size of the centered bounding box
        self.face_positioned_time_s = 2
        self.webcam_win_name = "User Camera"

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cima = cima

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7
        )

    def __draw_circle(self, color, center):
        """Draws a circle on the calibration window."""
        self.frame[:] = CV2_COLOR_BLACK
        cv2.circle(self.frame, center, self.radius, color, -1)
        cv2.circle(self.frame, center, self.radius_focus, CV2_COLOR_BLACK, -1)

    def __create_user_calib_window(self, screen_number=0):
        """Creates a calibration window in the top middle of the screen."""
        mon = monitor(screen_number)
        window_width, window_height = 300, 200

        top_middle_x = mon.info.x + (mon.info.width // 2) - (window_width // 2)
        top_middle_y = mon.info.y

        calib_window_name = 'Calibration'
        cv2.namedWindow(calib_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(calib_window_name, window_width, window_height)
        cv2.moveWindow(calib_window_name, top_middle_x, top_middle_y)

        self.frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        self.__draw_circle(CV2_COLOR_GREEN, (window_width // 2, self.radius))
        cv2.imshow(calib_window_name, self.frame)

    def __create_camera_window(self, screen_number=0):
        mon = monitor(screen_number)
        window_width, window_height = self.cap_width, self.cap_height

        win_pos_x = mon.info.x + (mon.info.width // 2) - (window_width // 2)
        win_pos_y = mon.info.y + (mon.info.height // 2) - (window_height // 2)

        cv2.namedWindow(self.webcam_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.webcam_win_name, window_width, window_height)
        cv2.moveWindow(self.webcam_win_name, win_pos_x, win_pos_y)

    def __get_face_center(self, frame):
        """Returns the midpoint coordinates between the two eyes of the first detected face."""
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                # Get keypoints for eyes
                keypoints = detection.location_data.relative_keypoints
                left_eye = keypoints[0]  # Left eye keypoint
                right_eye = keypoints[1]  # Right eye keypoint

                ih, iw, _ = frame.shape

                # Convert relative coordinates to absolute pixel values
                left_eye_x = int(left_eye.x * iw)
                left_eye_y = int(left_eye.y * ih)
                right_eye_x = int(right_eye.x * iw)
                right_eye_y = int(right_eye.y * ih)

                # Calculate midpoint between the two eyes
                midpoint_x = (left_eye_x + right_eye_x) // 2
                midpoint_y = (left_eye_y + right_eye_y) // 2

                return (midpoint_x, midpoint_y)
        return None

    def __draw_bbox(self, frame, bbox, color, thickness=2):
        """Draws a bounding box on the frame."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y - h//2), (x + w, y + h//2), color, thickness)

    def run(self, screen_number=0):
        """Runs the user calibration process with visual feedback."""
        valid = False
        countdown_started = False
        start_time = None
        self.__create_user_calib_window(screen_number)
        self.__create_camera_window(screen_number)

        self.cima.reset()
        while not valid:
            ret, webcam_frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam. Exiting...")
                break

            # Calculate centered bounding box
            h, w = webcam_frame.shape[:2]
            bbox = (
                w//2, h//2,
                self.bbox_size, self.bbox_size
            )

            # Get face position and draw elements
            face_center = self.__get_face_center(webcam_frame)
            in_bbox = False

            # Draw bounding box (green if face inside, red otherwise)
            box_color = CV2_COLOR_GREEN if countdown_started else CV2_COLOR_RED
            self.__draw_bbox(webcam_frame, bbox, box_color)

            if face_center:
                # Draw face center point
                cv2.circle(webcam_frame, face_center, 5, CV2_COLOR_RED, -1)

                # Check if face center is within bbox
                x, y, w, h = bbox
                in_bbox = (x <= face_center[0] <= x + w and
                           y - h//2 <= face_center[1] <= y + h//2)

            # Countdown logic
            if in_bbox:
                if not countdown_started:
                    start_time = time.time()
                    countdown_started = True
                else:
                    elapsed = time.time() - start_time
                    if elapsed >= self.face_positioned_time_s:
                        if self.cima.eyes_analyzer.check_for_sufficient_gaze_detection(webcam_frame):
                            valid = True
                    else:
                        # Display countdown
                        remaining = self.face_positioned_time_s - int(elapsed)
                        cv2.putText(
                            webcam_frame, f"Hold position... {remaining}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, CV2_COLOR_GREEN, 2
                        )
            else:
                self.cima.reset()
                countdown_started = False
                cv2.putText(
                    webcam_frame, "Align your face within the box",
                    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CV2_COLOR_GREEN, 2
                )

            cv2.imshow(self.webcam_win_name, webcam_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                print("Calibration aborted by user.")
                break

        cv2.destroyAllWindows()
        return valid
