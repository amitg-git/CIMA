import cv2

from config import Config
from gui.UserCalib import UserCalib
from helper.file import *
from helper.helper import *
from helper.monitor import *
from cima.cima import CIMA
from record.webcam import webcam


class CI_Screen:
    def __init__(self, test_name: str, distance_mm, positions, cima: CIMA):
        self.cima = cima

        # Create statistics instance
        self.test_name = test_name
        self.logger = csv_logger(self.test_name, self.cima.user_id)

        # Initialize video capturing
        self.webcam = webcam(self.cima.user_id, self.test_name + "-video")

        # Initialize Logger
        self.positions = positions
        self.distance_mm = distance_mm
        self.current_dist_mm = logvar(['d'], ['mm'])
        self.current_dist_mm.d = 0
        self.logger.add_log(self.current_dist_mm)
        self.dotL = scr_point(0, 0, "pixels", "Left")
        self.dotR = scr_point(0, 0, "pixels", "Right")
        self.logger.add_log(self.dotL)
        self.logger.add_log(self.dotR)

        self.log = logvar(
            ['valid'],
            ['']
        )
        self.logger.add_log(self.log)
        self.log.valid = 0

        # Initialize Test
        self.radius_focus = 8
        self.radius = 25
        self.current_dist = 0
        self.cur_pos = 0
        self.state = TEST_IDLE
        self.screen_width = None
        self.screen_height = None
        self.frame = None
        self.centerL = None
        self.centerR = None
        self.left_clicked = False
        self.right_clicked = False
        self.mon_inst = monitor()

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
        cv2.circle(self.frame, self.centerL, self.radius, color, -1)
        cv2.circle(self.frame, self.centerL, self.radius_focus, CV2_COLOR_BLACK, -1)
        cv2.circle(self.frame, self.centerR, self.radius, color, -1)
        cv2.circle(self.frame, self.centerR, self.radius_focus, CV2_COLOR_BLACK, -1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_clicked = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_clicked = True

    def handle_mouse_clicked(self):
        if self.right_clicked:
            # Abort position
            self.cur_pos += 1
            self.current_dist = 0
            self.right_clicked = False
        elif self.left_clicked:
            # Next position
            self.state += 1  # next state

            if self.state == TEST_END:
                # Clear the frame and move to the next position after second click
                self.frame[:] = CV2_COLOR_BLACK

                self.current_dist += 1
                if self.current_dist >= len(self.distance_mm):
                    self.current_dist = 0
                    self.cur_pos += 1

                self.state = TEST_IDLE

            # Reset the mouse clicked flag
            self.left_clicked = False

    def update_circle_pos(self):
        x_ratio, y_ratio = self.positions[self.cur_pos]
        mid = scr_point(int(x_ratio * self.screen_width),
                        int(y_ratio * self.screen_height))

        distance = self.mon_inst.width_mm2pixels(self.distance_mm)  # unit [pixels]
        self.centerL = (int(mid.x - 0.5 * distance[self.current_dist]), int(mid.y))
        self.centerR = (int(mid.x + 0.5 * distance[self.current_dist]), int(mid.y))

        # Update log
        self.dotL.set(self.centerL[0], self.centerL[1])
        self.dotR.set(self.centerR[0], self.centerR[1])
        self.current_dist_mm.d = self.distance_mm[self.current_dist]

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
            text = f"Distance: {self.distance_mm[self.current_dist]} mm, #dot: {self.cur_pos}"
            cv2.putText(self.frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV2_COLOR_WHITE, 2)
            cv2.imshow(self.test_name, self.frame)

            # Handle mouse click logic
            self.handle_mouse_clicked()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()
        self.webcam.stop_recording()
        return True
