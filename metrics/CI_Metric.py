import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import config
from gui.UserCalib import UserCalib
from helper.file import get_unique_filename
from helper.helper import *
from helper.monitor import *
from cima.cima import CIMA
from helper.statistics import CI_Classification, RealtimeLogger


class ci_metric:
    def __init__(self, cima: CIMA, thresholds, screen_number=0):
        self.test_name = "ci_metric"
        self.cima = cima
        self.screen_number = screen_number
        self.thresholds = thresholds

        # Convergence test plan
        self.focal_area_index = [3, 4, 5]
        self.positions = [
            (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
        ]
        distance_mm_range = list(range(60, 90, 5))
        self.distance_mm = self._ci_generate_samples(distance_mm_range, num_values=3, num_zeros=1)
        # self.distance_mm = np.array([60, 0, 80])

        self.state = TEST_IDLE
        self.overall_preds = {}
        self.preds = []

        # Initialize Test
        self.timeout = config.metric.CI_METRIC_SUBTEST_DURATION_FRAMES  # unit [frames]
        self.radius_focus = 8
        self.radius = 25
        self.cur_dist = 0
        self.cur_pos = 0
        self.screen_width = None
        self.screen_height = None
        self.frame = None
        self.centerL = None
        self.centerR = None
        self.left_clicked = False
        self.right_clicked = False
        self.mon = monitor(self.screen_number)
        self.test_logger = None
        self.test_logger_valid: int = 0

    def close(self):
        pass

    @staticmethod
    def _ci_generate_samples(dist_range, *, num_values, num_zeros):
        values = random.sample(dist_range, num_values)
        samples = values + [0] * num_zeros
        random.shuffle(samples)
        return np.array(samples)

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
        if self.right_clicked and self.state == TEST_IDLE:
            # Abort position
            self.cur_pos += 1
            self.cur_dist = 0
            self.right_clicked = False
        elif self.left_clicked:
            # Next position
            self.state += 1  # next state

            if self.state == TEST_END:
                # Clear the frame and move to the next position after second click
                self.frame[:] = CV2_COLOR_BLACK

                self.cur_dist += 1
                if self.cur_dist >= len(self.distance_mm):
                    self.cur_dist = 0
                    self.cur_pos += 1

                self.state = TEST_IDLE

            # Reset the mouse clicked flag
            self.left_clicked = False

    def update_circle_pos(self):
        x_ratio, y_ratio = self.positions[self.cur_pos]
        mid = scr_point(int(x_ratio * self.screen_width),
                        int(y_ratio * self.screen_height))

        distance = self.mon.width_mm2pixels(self.distance_mm)  # unit [pixels]
        self.centerL = (int(mid.x - 0.5 * distance[self.cur_dist]), int(mid.y))
        self.centerR = (int(mid.x + 0.5 * distance[self.cur_dist]), int(mid.y))

    def analyze_frame(self, frame):
        self.cima.eyes_analyzer.analyze_frame(frame)

    def run(self, calib=True, save_log:bool = False):
        if calib:
            # Gaze pre-calibration
            user_calib = UserCalib(self.cima)
            if not user_calib.run(self.screen_number):
                return False

        # save predictions to a file
        if save_log:
            today = time.strftime('%Y%m%d')
            self.test_logger = RealtimeLogger(
                get_unique_filename(f"user_reports/{self.cima.user_id}_{today}_metric_1.csv")
            )

        # Open camera
        cap = cv2.VideoCapture(0)

        # Run the test
        self.cima.eyes_analyzer.reset_internal_data()
        self.create_fullscreen_window()
        cv2.setMouseCallback(self.test_name, self.mouse_callback)
        frame_count = 0

        while self.cur_pos < len(self.positions):
            _, frame = cap.read()
            self.analyze_frame(frame)

            # Update position and color based on state
            if self.state == TEST_IDLE:
                self.update_circle_pos()
                self.draw_circle(CV2_COLOR_WHITE)
                self.preds = []
                frame_count = 0

                self.ci_classifier = CI_Classification(
                    window_size=self.timeout,
                    value_for_true=self.thresholds[self.focal_area_index[self.cur_pos]],
                    threshold_for_detect=0.4
                )
                self.test_logger_valid = 0
            elif self.state == TEST_START:
                ci_dist = self.cima.eyes_analyzer.ci_dist.get_last()
                self.ci_classifier.add(ci_dist)

                frame_count += 1
                self.draw_circle(CV2_COLOR_RED)
                if frame_count >= self.timeout:
                    self.overall_preds[(self.cur_pos, self.cur_dist, self.distance_mm[self.cur_dist])] = self.ci_classifier.get_all()
                    self.left_clicked = True
                self.test_logger_valid = 1
            else:
                self.test_logger_valid = 0

            # Update test's frame
            d_index, dist_mm, cur_pos = self.cur_dist, self.distance_mm, self.cur_pos
            text = f"Distance: {dist_mm[d_index]} mm ({d_index}/{len(dist_mm)}), #pos: {cur_pos} "
            cv2.putText(self.frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV2_COLOR_WHITE, 2)

            ci_dist = int(round(self.cima.eyes_analyzer.ci_dist.get()))
            threshold = self.thresholds[self.focal_area_index[cur_pos]]
            sign = ">" if ci_dist > threshold else "<="
            color = CV2_COLOR_RED if ci_dist > threshold else CV2_COLOR_BLUE
            text = f"SPDT: {ci_dist} {sign} {threshold}"
            cv2.putText(self.frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            self.test_logger.add({
                'scr_dist[cm]': self.cima.eyes_analyzer.dc.get('scr_dist'),
                'SPCD_real[mm]': dist_mm[d_index],
                'SPCD_est[mm]': ci_dist,
                'threshold[mm]': threshold,
                'focal_area_index[]': self.focal_area_index[cur_pos],
                'valid[]': self.test_logger_valid,
            })

            cv2.imshow(self.test_name, self.frame)

            # Handle mouse click logic
            self.handle_mouse_clicked()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break

        # clean and exit
        self.test_logger.save_stat()
        cv2.destroyAllWindows()

        return True

    @staticmethod
    def classify_ci(preds):
        n = len(preds)
        if n == 0:
            return None

        n_ci = sum(preds != 0)
        if n_ci >= 0.4 * n:
            return True
        return False

    def plot_results(self):
        positions = set(pos for (pos, n, dist) in self.overall_preds.keys())
        for pos in positions:
            distances = self.distance_mm
            n_values = list(range(len(self.distance_mm)))
            correct_counts = []
            incorrect_counts = []
            accs = []

            # This part is simplified for clarity, your original logic is fine.
            for n, dist in zip(n_values, distances):
                preds = self.overall_preds.get((pos, n, dist), [])
                correct_label = 0 if dist == 0 else 1
                correct = sum(1 for p in preds if p == correct_label)
                incorrect = len(preds) - correct
                acc = correct / len(preds) if len(preds) != 0 else 0
                correct_counts.append(correct)
                incorrect_counts.append(incorrect)
                accs.append(acc)

            x = np.arange(len(distances))
            bar_width = 0.6

            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x, correct_counts, bar_width, label='Correct', color='blue')
            bars2 = ax.bar(x, incorrect_counts, bar_width, bottom=correct_counts, label='Incorrect', color='red')

            #  Adds a 10% margin to the top of the y-axis.
            ax.margins(y=0.1)

            # Annotate accuracy above each bar
            for idx, acc in enumerate(accs):
                height = correct_counts[idx] + incorrect_counts[idx]

                # Use the bar's x position for more reliable centering
                bar_x_center = bars1[idx].get_x() + bars1[idx].get_width() / 2

                text_offset = ax.get_ylim()[1] * 0.01  # 1% of the new y-axis height
                ax.text(bar_x_center, height + text_offset, f"{acc:.2f}", ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

            ax.set_xlabel('SPCD[mm]')
            ax.set_ylabel('Prediction Count')
            ax.set_title(f'Correct and Incorrect Predictions per Distance\nPosition: {pos}', y=1.05)
            ax.set_xticks(x)
            ax.set_xticklabels([str(d) for d in distances])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Use bbox_inches='tight' with tight_layout for better legend placement
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust rect to make space for legend
            plt.show()
