import cv2

import config
from helper.helper import *
from helper.monitor import *
from cima.cima import CIMA
from gui.UserCalib import UserCalib


class CI_UserCalib:
    def __init__(self, cima: CIMA, test_name=None, screen_number=0):
        self.cima = cima
        self.screen_number = screen_number

        # Create statistics instance
        self.test_name = "ci_calib" if test_name is None else test_name

        # Initialize Test
        self.timeout = config.analyze.CI_THRESHOLD_CALIB_DURATION
        self.overall_preds = {}
        self.preds = []
        self.radius_focus = 8
        self.radius = 25
        self.positions = [
            (0.2, 0.1), (0.5, 0.1), (0.8, 0.1),
            (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
            (0.2, 0.9), (0.5, 0.9), (0.8, 0.9)
        ]
        self.cur_pos = 0
        self.state = TEST_IDLE
        self.screen_width = None
        self.screen_height = None
        self.frame = None
        self.center = None
        self.left_clicked = False
        self.mon = monitor(self.screen_number)

    def close(self):
        pass

    def create_fullscreen_window(self):
        cv2.namedWindow(self.test_name, cv2.WINDOW_NORMAL)

        # Move window to the selected screen
        cv2.moveWindow(self.test_name, self.mon.info.x, self.mon.info.y)

        # Set the window to fullscreen
        cv2.setWindowProperty(self.test_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get the screen dimensions
        self.screen_width = self.mon.info.width
        self.screen_height = self.mon.info.height

        # Create a black frame with the screen dimensions
        self.frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def draw_circle(self, color):
        self.frame[:] = CV2_COLOR_BLACK
        cv2.circle(self.frame, self.center, self.radius, color, -1)
        cv2.circle(self.frame, self.center, self.radius_focus, CV2_COLOR_BLACK, -1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_clicked = True

    def handle_mouse_clicked(self, skip):
        if self.left_clicked:
            # Next position
            self.state += 1  # next state

            if self.state == TEST_END:
                # Clear the frame and move to the next position after second click
                self.frame[:] = CV2_COLOR_BLACK

                if self.cur_pos in skip:
                    self.overall_preds[self.cur_pos] = [0, ]
                else:
                    self.overall_preds[self.cur_pos] = self.preds

                self.cur_pos += 1
                self.state = TEST_IDLE

            # Reset the mouse clicked flag
            self.left_clicked = False

    def update_circle_pos(self):
        x_ratio, y_ratio = self.positions[self.cur_pos]
        self.center = (int(x_ratio * self.screen_width),
                       int(y_ratio * self.screen_height))

    def get_focal_area_preds(self):
        return self.overall_preds

    def analyze_frame(self, frame):
        self.cima.eyes_analyzer.analyze_frame(frame)

    def run(self, calib=True, skip=[]):
        if calib:
            # Gaze pre-calibration
            user_calib = UserCalib(self.cima)
            if not user_calib.run(self.screen_number):
                return False

        # Open camera
        cap = cv2.VideoCapture(0)

        # Run the test
        self.cima.eyes_analyzer.reset_internal_data()
        self.create_fullscreen_window()
        cv2.setMouseCallback(self.test_name, self.mouse_callback)
        frame_count = 0

        abort_flag = False
        while self.cur_pos < len(self.positions):
            _, frame = cap.read()
            self.analyze_frame(frame)

            # Update position and color based on state
            if self.state == TEST_IDLE:
                self.update_circle_pos()
                self.draw_circle(CV2_COLOR_WHITE)
                self.preds = []
                frame_count = 0
            elif self.state == TEST_START:
                self.draw_circle(CV2_COLOR_RED)
                self.preds.append(self.cima.eyes_analyzer.ci_dist.get_last())
                frame_count += 1
                if frame_count >= self.timeout or self.cur_pos in skip:
                    self.left_clicked = True

            # Update test's frame
            cv2.imshow(self.test_name, self.frame)

            # Handle mouse click logic
            self.handle_mouse_clicked(skip)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                abort_flag = True
                break

        cv2.destroyAllWindows()
        return not abort_flag
