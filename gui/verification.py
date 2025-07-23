import tkinter as tk

import numpy as np
from tkinterdnd2 import TkinterDnD

import config
from config import Config
from gui.CI_UserCalib import CI_UserCalib
from helper.monitor import get_laptop_monitor_number
from cima.cima import CIMA
from metrics.CI_Metric import ci_metric
from metrics.Face_Metric import face_metric


class verification_screen:
    def __init__(self, prev_screen, cima: CIMA):
        self.content_frame = None
        self.menu_frame = None
        self.gaze_calib_flag = None

        self.prev_screen = prev_screen
        self.master = TkinterDnD.Tk()
        self.cima = cima

        self.master.title(Config.APP_TITLE)
        self.center_window("verification")
        self.setup_ui()

        # Set up the window close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Get laptop monitor number:
        self.screen_number = get_laptop_monitor_number()

    def on_closing(self):
        self.master.destroy()
        self.prev_screen.deiconify()  # Show the prev screen again

    def center_window(self, screen_type):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        geometry = Config.get_screen_geometry(screen_type)
        width, height = map(int, geometry.split('x'))
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.master.geometry(f'{geometry}+{x}+{y}')

    def setup_ui(self):
        # Create a frame for the menu
        self.menu_frame = tk.Frame(self.master, width=Config.MENU_WIDTH, bg='lightgray')
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.menu_frame.pack_propagate(False)  # Prevent the frame from shrinking

        # Create menu buttons
        tk.Button(self.menu_frame, text="Face metric", command=self.face_metric_screen).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="CI metric", command=self.ci_metric_screen).pack(fill=tk.X, pady=10)
        # Add spacer
        tk.Frame(self.menu_frame, height=20, bg='lightgray').pack(fill=tk.X, expand=True)

        # Create a frame for the content
        self.content_frame = tk.Frame(self.master)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Flag for Camera
        self.gaze_calib_flag = tk.BooleanVar(master=self.content_frame, value=True)
        checkbox_entry = tk.Checkbutton(self.content_frame, text="Gaze\nCalibration", variable=self.gaze_calib_flag)
        checkbox_entry.place(x=5, y=50)

        # Welcome message
        tk.Label(self.content_frame, text=f"Welcome, User {self.cima.user_id}", font=("Arial", 24)).pack(pady=20)

    def __auto_ci_calib_process(self):
        threshold_valid_flag = False
        abort_flag = False
        focal_area_ci_threshold = []

        while not threshold_valid_flag and not abort_flag:
            # Calibrate CI thresholds for each focal area
            ci_usercalib = CI_UserCalib(self.cima, screen_number=self.screen_number)
            calib_success = ci_usercalib.run(calib=self.gaze_calib_flag.get(), skip=[0, 1, 2, 6, 7, 8])
            focal_area_ci_threshold = {}
            if calib_success:
                focal_area_preds = ci_usercalib.get_focal_area_preds()
                # Set Thresholds
                print("Statistics for preds:")
                for i, (key, preds) in enumerate(focal_area_preds.items()):
                    p_min, p_max, p_90, p_mean, p_median, p_std = \
                        np.min(preds), np.max(preds), np.round(np.percentile(preds, 90), 2), np.mean(preds), np.median(preds), np.std(preds)
                    print(
                        f"{i}: Min: {p_min:.2f}, Max: {p_max:.2f}, p90: {p_90}, Med: {p_median:.2f}, Mean: {p_mean:.2f}, Std: {p_std:.2f}",
                        end='')
                    focal_area_ci_threshold[key] = \
                        np.round(max(config.analyze.MIN_CI_THRESHOLD, (p_90 * 1.1)), 1)

                    print(f"\tthrehold: {focal_area_ci_threshold[key]}")

                if all(x <= config.analyze.MAX_CI_THRESHOLD for x in focal_area_ci_threshold.values()):
                    threshold_valid_flag = True
            else:
                abort_flag = True

        if not abort_flag and threshold_valid_flag:
            return focal_area_ci_threshold
        else:
            return None

    def ci_metric_screen(self):
        self.master.withdraw()  # Hide the main screen

        focal_area_ci_threshold = self.__auto_ci_calib_process()
        if focal_area_ci_threshold is not None:
            # Create CI metric instance and run the test
            metric = ci_metric(self.cima, focal_area_ci_threshold, screen_number=self.screen_number)
            if metric.run(calib=False, save_log=True):
                metric.plot_results()
            metric.close()

        # Metric is finished
        self.master.deiconify()  # Show the prev screen again

    def face_metric_screen(self):
        self.master.withdraw()  # Hide the main screen
        metric = face_metric(self.cima,
                             test_duration=config.metric.FACE_METRIC_TEST_DURATION_SEC,
                             screen_number=self.screen_number)
        if metric.run():
            metric.plot_results()
        metric.close()
        self.master.deiconify()  # Show the prev screen again
