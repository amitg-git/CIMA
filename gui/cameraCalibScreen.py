import cv2
import numpy as np
from datetime import datetime

from helper.helper import *
from helper.monitor import monitor
from camera_calibration.camera_calib import video_to_images, camera_calib


class cameraCalibScreen:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.webcam_win_name = "User Camera"

        self.calib_images_path = "camera_calibration/images/"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.cap_name = f"camera_calibration/videos/camera_calib_{timestamp}.mp4"

    def __create_user_calib_window(self, screen_number=0):
        """Creates a calibration window in the top middle of the screen."""
        mon = monitor(screen_number)
        window_width, window_height = 300, 200

        top_middle_x = mon.info.x + (mon.info.width // 2) - (window_width // 2)
        top_middle_y = mon.info.y

        calib_window_name = 'Camera Calibration'
        cv2.namedWindow(calib_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(calib_window_name, window_width, window_height)
        cv2.moveWindow(calib_window_name, top_middle_x, top_middle_y)

        frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Add instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = CV2_COLOR_WHITE
        thickness = 2

        text1 = "Press 's' to start"
        text2 = "Press 'q' to quit"

        # Calculate text size and position for centering
        (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
        (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)

        # Position text1 near the top center
        text1_x = (window_width - text1_width) // 2
        text1_y = (window_height // 2) - 20

        # Position text2 below text1
        text2_x = (window_width - text2_width) // 2
        text2_y = (window_height // 2) + 20

        cv2.putText(frame, text1, (text1_x, text1_y), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow(calib_window_name, frame)

    def wait_screen(self):
        key = None
        while key != ord('s') and key != ord('q'):
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame")
                continue

            # show frame
            cv2.imshow(self.webcam_win_name, frame)

            key = cv2.waitKey(10) & 0xFF
        cv2.destroyAllWindows()

        # Check of exit
        if key == ord('q'):
            print("Calibration aborted by user.")
            return 'quit'
        return 'start'

    def record_screen(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.cap_name, fourcc, self.cap_fps,
                                       (self.cap_width, self.cap_height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame")
                continue

            # show frame
            cv2.imshow(self.webcam_win_name, frame)
            video_writer.write(frame)

            # handle APP exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or \
                    key == ord('q') or \
                    cv2.getWindowProperty(self.webcam_win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        video_writer.release()
        cv2.destroyAllWindows()

    def run(self, screen_number, dump_file):
        self.__create_user_calib_window(screen_number)

        if self.wait_screen() == 'quit':
            return -1

        self.record_screen()
        video_to_images(self.cap_name, "calib", self.calib_images_path)
        return camera_calib(self.calib_images_path,
                            frame_to_skip=5, max_images=100,
                            dump_file=dump_file)
