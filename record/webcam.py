import cv2
from config import Config
from helper.file import *


class webcam:
    def __init__(self, user_id, window_name: str):
        self.frame = None
        self.logger = None
        self.cap_name = None
        self.video_writer = None
        self.user_id = user_id
        self.window_name = window_name

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.ms = 1000 // self.fps

    def init_recording_files(self, file_name: str, logger: csv_logger):
        if not Config.save_webcam_file:
            return

        self.logger = logger
        self.logger.open()

        # Initialize video writer and Start recording
        if self.video_writer:
            self.video_writer.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.cap_name = get_unique_filename(f"captures/{self.user_id}_{file_name}_1.mp4")
        self.video_writer = cv2.VideoWriter(self.cap_name, fourcc, self.fps,
                                            (self.frame_width, self.frame_height))

    def read(self):
        _, self.frame = self.cap.read()

    def save_frame_and_log(self):
        self.video_writer.write(self.frame)
        self.logger.push()

    def show_frame(self):
        cv2.imshow(self.window_name, self.frame)

    def stop_recording(self):
        if self.cap.isOpened():
            self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        self.logger.close()
