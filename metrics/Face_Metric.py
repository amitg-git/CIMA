import cv2
from matplotlib import pyplot as plt, gridspec

from gui.UserCalib import UserCalib
from helper.file import *
from helper.helper import *
from helper.monitor import *
from cima.cima import CIMA


class face_metric:
    def __init__(self, cima: CIMA, test_duration, screen_number=0):
        self.test_name = "face_metric"
        self.screen_number = screen_number
        self.cima = cima

        self.state = TEST_IDLE
        self.face_detections = []
        self.eyes_open = []

        # Initialize Test
        self.test_duration = test_duration  # unit [sec]
        self.left_clicked = False
        self.right_clicked = False
        self.mon = monitor(self.screen_number)

    def close(self):
        pass

    def create_window(self):
        cv2.namedWindow(self.test_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.test_name, self.mon.info.x, self.mon.info.y)
        cv2.setWindowProperty(self.test_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_clicked = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_clicked = True

    def handle_mouse_clicked(self):
        if self.left_clicked and self.state == TEST_IDLE:
            self.state = TEST_START
            self.left_clicked = False
        if self.right_clicked and self.state == TEST_START:
            self.state = TEST_END
            self.right_clicked = False

    def analyze_frame(self, frame):
        is_face_detect, is_eyes_open, _ = self.cima.eyes_analyzer.analyze_frame(frame)
        self.face_detections.append(is_face_detect)
        self.eyes_open.append(is_eyes_open)

    def run(self):
        # Open camera
        cap = cv2.VideoCapture(0)

        # Run the test
        self.cima.eyes_analyzer.reset_internal_data()
        self.create_window()
        cv2.setMouseCallback(self.test_name, self.mouse_callback)

        # timing init
        start_time = time.time()

        while cap.isOpened():
            # read one frame from camera
            success, frame = cap.read()
            if not success:
                continue

            if self.state == TEST_IDLE:
                start_time = time.time()
            elif self.state == TEST_START:
                self.analyze_frame(frame)
                # self.cima.show_ear_param_landmarks(frame)

                if time.time() - start_time > self.test_duration:
                    break

            elif self.state == TEST_END:
                break

            cv2.imshow(self.test_name, frame)

            # Handle mouse click logic
            self.handle_mouse_clicked()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()
        return True

    def plot_results(self):
        # Data preparation
        face_true = sum(self.face_detections)
        face_false = len(self.face_detections) - face_true
        eyes_true = sum(self.eyes_open)
        eyes_false = len(self.eyes_open) - eyes_true

        # Accuracy
        face_acc = face_true / len(self.face_detections) if self.face_detections else 0
        eyes_acc = eyes_true / len(self.eyes_open) if self.eyes_open else 0

        # Bar positions
        x = np.arange(2)
        bar_width = 0.6

        # Create custom grid layout: 2 columns, right column split into 2 rows
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

        # --- Left: Stacked Bar Plot (spans both rows) ---
        ax_bar = fig.add_subplot(gs[:, 0])
        ax_bar.bar(x[0], face_true, bar_width, label='Face Detected', color='green')
        ax_bar.bar(x[0], face_false, bar_width, bottom=face_true, label='Face Not Detected', color='gray')
        ax_bar.bar(x[1], eyes_true, bar_width, label='Eyes Open', color='cyan')
        ax_bar.bar(x[1], eyes_false, bar_width, bottom=eyes_true, label='Eyes Closed', color='orange')

        y_pos = face_true - 0.05 * (face_true + face_false)
        ax_bar.text(x[0], y_pos, f"{face_acc:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

        y_pos = eyes_true - 0.05 * (eyes_true + eyes_false)
        ax_bar.text(x[1], y_pos, f"{eyes_acc:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

        labels = ['Face Detected', 'Eyes Open']
        ax_bar.get_xaxis().set_visible(False)
        for i, label in enumerate(labels):
            ax_bar.text(x[i], -0.5, label, ha='center', va='top', fontsize=12)
        ax_bar.set_ylabel('Count')
        ax_bar.set_title('Face Detection and Eyes Open Results')
        ax_bar.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Top Right: Face Detection per Frame ---
        ax_face = fig.add_subplot(gs[0, 1])
        frames = np.arange(len(self.face_detections))
        ax_face.plot(frames, self.face_detections, label='Face Detected (1/0)', color='green', linewidth=1)
        ax_face.set_ylim(-0.1, 1.1)
        ax_face.set_xlabel('Frame Number')
        ax_face.set_ylabel('Status')
        ax_face.set_title('Face Detection Status per Frame')
        ax_face.legend()
        ax_face.grid(True, linestyle='--', alpha=0.7)

        # --- Bottom Right: Eyes Status per Frame ---
        ax_eyes = fig.add_subplot(gs[1, 1])
        ax_eyes.plot(frames, self.eyes_open, label='Eyes Open (1/0)', color='cyan', linewidth=1)
        ax_eyes.set_ylim(-0.1, 1.1)
        ax_eyes.set_xlabel('Frame Number')
        ax_eyes.set_ylabel('Status')
        ax_eyes.set_title('Eyes Status per Frame')
        ax_eyes.legend()
        ax_eyes.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
