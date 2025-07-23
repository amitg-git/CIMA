import cv2

from config import Config
from helper.file import *
from helper.helper import *
from helper.monitor import *
from cima.cima import CIMA
from record.webcam import webcam
from gui.UserCalib import UserCalib


class movingDotScreen:
    def __init__(self, cima: CIMA, positions, test_name=None):
        self.cima = cima

        # Create statistics instance
        self.test_name = "moving-dot" if test_name is None else test_name
        self.logger = csv_logger(self.test_name, self.cima.user_id)

        # Initialize video capturing
        self.webcam = webcam(self.cima.user_id, window_name=self.test_name + "-video")

        # Initialize Logger
        self.dot = scr_point(0, 0, "pixels")
        self.logger.add_log(self.dot)
        self.log = logvar(
            ['valid'],
            ['']
        )
        self.logger.add_log(self.log)
        self.log.valid = 0

        # Initialize Test
        self.radius_focus = 8
        self.radius = 25
        self.positions = positions
        self.cur_pos = 0
        self.state = TEST_IDLE
        self.screen_width = None
        self.screen_height = None
        self.frame = None
        self.center = None
        self.left_clicked = False

    def close(self):
        self.logger.close()

    def get_csv_filename(self):
        return self.logger.filename

    def get_video_filename(self):
        return self.webcam.cap_name

    def create_fullscreen_window(self, screen_number=0):
        # Get the selected monitor
        mon = monitor(screen_number)

        cv2.namedWindow(self.test_name, cv2.WINDOW_NORMAL)

        # Move window to the selected screen
        cv2.moveWindow(self.test_name, mon.info.x, mon.info.y)

        # Set the window to fullscreen
        cv2.setWindowProperty(self.test_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get the screen dimensions
        self.screen_width = mon.info.width
        self.screen_height = mon.info.height

        # Create a black frame with the screen dimensions
        self.frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def draw_circle(self, color):
        self.frame[:] = CV2_COLOR_BLACK
        cv2.circle(self.frame, self.center, self.radius, color, -1)
        cv2.circle(self.frame, self.center, self.radius_focus, CV2_COLOR_BLACK, -1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_clicked = True

    def handle_mouse_clicked(self):
        if self.left_clicked:
            # Next position
            self.state += 1  # next state

            if self.state == TEST_END:
                # Clear the frame and move to the next position after second click
                self.frame[:] = CV2_COLOR_BLACK
                self.cur_pos += 1
                self.state = TEST_IDLE

            # Reset the mouse clicked flag
            self.left_clicked = False

    def update_circle_pos(self):
        x_ratio, y_ratio = self.positions[self.cur_pos]
        self.center = (int(x_ratio * self.screen_width),
                       int(y_ratio * self.screen_height))

        # Update log
        self.dot.set(self.center[0], self.center[1])

    def run(self, screen_number=0, calib=True):
        if calib:
            # Gaze pre-calibration
            user_calib = UserCalib(self.cima)
            if not user_calib.run(screen_number):
                return False

        # Run the test
        self.webcam.init_recording_files(self.test_name, self.logger)
        self.create_fullscreen_window(screen_number)
        cv2.setMouseCallback(self.test_name, self.mouse_callback)
        while self.cur_pos < len(self.positions):
            # Get webcam frame
            self.webcam.read()

            # Update position and color based on state
            if self.state == TEST_IDLE:
                self.update_circle_pos()
                self.draw_circle(CV2_COLOR_WHITE)
                self.log.valid = 0
            elif self.state == TEST_START:
                self.draw_circle(CV2_COLOR_RED)
                self.log.valid = 1
            else:
                self.log.valid = 0

            # Save webcam frame and log, show webcam if need
            self.webcam.save_frame_and_log()
            if Config.VIEW_CAM_AT_TEST:
                self.webcam.show_frame()

            # Update test's frame
            cv2.imshow(self.test_name, self.frame)

            # Handle mouse click logic
            self.handle_mouse_clicked()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()
        self.webcam.stop_recording()
        return True
