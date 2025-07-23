import tkinter as tk
from tkinter import messagebox
from datetime import datetime

import numpy as np

import config
from gui.CI_UserCalib import CI_UserCalib
from helper.monitor import get_laptop_monitor_number
from gui.UserCalib import UserCalib
from user_manager import UserManager
from config import Config

from gui.calibration import calib_screen
from gui.verification import verification_screen
from gui.csv_analyzer import csv_analyzer_screen
from gui.video_analyzer import video_analyzer_screen
from cima.cima import CIMA


def restore_placeholder(event, entries, placeholder_texts):
    for entry, placeholder_text in zip(entries, placeholder_texts):
        if event.widget != entry and entry.get() == "":
            entry.insert(0, placeholder_text)


def manual_ci_threshold_visible(flag, entries, place_infos, *args):
    if flag.get():
        for entry in entries:
            entry.place_forget()
    else:
        for entry, place_info in zip(entries, place_infos):
            entry.place(**place_info)


class LoginScreen:
    def __init__(self, master: tk.Tk):
        self.id_entry = None
        self.cima = CIMA()
        self.user_manager = UserManager()

        self.master = master
        self.master.title(Config.APP_TITLE)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)  # Set up the window close event

        self.center_window("login")

        self.visualize_flag = None

        self.setup_ui()

    def on_closing(self):
        self.master.quit()
        self.master.destroy()

    def center_window(self, screen_type):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        geometry = Config.get_screen_geometry(screen_type)
        width, height = map(int, geometry.split('x'))
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.master.geometry(f'{geometry}+{x}+{y}')

    def setup_ui(self):
        # Project title
        title_label = tk.Label(self.master, text="CIMA", font=("Arial", 24))
        title_label.pack(pady=20)

        # User ID input
        self.id_entry = tk.Entry(self.master, font=("Arial", 14))
        self.id_entry.pack(pady=10)

        # Submit button
        submit_id_button = tk.Button(self.master, text="Submit userID",
                                     command=lambda: self.process_user_id(self.id_entry.get().strip()))
        submit_id_button.pack(pady=10)

        if not Config.IS_PRODUCTION:
            submit_date_button = tk.Button(self.master, text="Auto submit Date",
                                           command=lambda: self.process_user_id(datetime.now().strftime("%Y%m%d")))
            submit_date_button.pack(pady=10)

        # Bind the Enter key to the process_user_id method
        self.master.bind('<Return>', self.on_enter)

    def on_enter(self, event):
        self.process_user_id(self.id_entry.get().strip())

    def process_user_id(self, user_id):
        if not user_id:
            messagebox.showerror("Error", "Please enter a user ID")
            return

        if self.user_manager.user_exists(user_id):
            self.show_next_screen(user_id)
        else:
            self.user_manager.add_user(user_id)
            messagebox.showinfo("New User", f"User ID {user_id} has been added.")
            self.show_next_screen(user_id)

    def show_next_screen(self, user_id):
        self.cima.user_id = user_id  # Set the user_id in the CIMA instance
        MenuScreen(self, self.cima)


class MenuScreen:
    def __init__(self, login_screen: LoginScreen, cima: CIMA):
        self.menu_frame = None
        self.content_frame = None
        self.visualize_flag = None
        self.gaze_calib_flag = None
        self.ci_calib_flag = None
        self.ci_threshold_entry = None
        self.di_threshold_entry = None
        self.decision_window_size_entry = None
        self.proportion_entry = None
        self.cima = cima

        login_screen.master.withdraw()  # Hide the prev screen
        self.login_screen = login_screen

        self.master = tk.Tk()
        self.master.title(Config.APP_TITLE)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)  # Set up the window close event

        self.center_window("menu")
        self.setup_ui()

    def on_closing(self):
        self.master.quit()
        self.master.destroy()

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
        tk.Button(self.menu_frame, text="Calibrations", command=self.open_calib_screen).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="CSV Analyzer", command=self.open_csv_analyzer_screen).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="Video Analyzer", command=self.open_video_analyzer_screen).pack(fill=tk.X,
                                                                                                        pady=10)
        tk.Button(self.menu_frame, text="Camera", command=self.open_camera).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="Verification", command=self.open_verification).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="Run In Background", command=self.run_in_background).pack(fill=tk.X, pady=10)

        # Add spacer
        tk.Frame(self.menu_frame, height=20, bg='lightgray').pack(fill=tk.X, expand=True)

        # Add Log Out button at the bottom
        tk.Button(self.menu_frame, text="Log Out", command=self.log_out).pack(fill=tk.X, pady=10, side=tk.BOTTOM)

        # Create a frame for the content
        self.content_frame = tk.Frame(self.master)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Welcome message
        tk.Label(self.content_frame, text=f"Welcome, User {self.cima.user_id}", font=("Arial", 24)).pack(pady=20)

        # Flag for visualize the video analysis
        self.visualize_flag = tk.BooleanVar(master=self.content_frame, value=True)
        checkbox_entry = tk.Checkbutton(self.content_frame, text="Visualize", variable=self.visualize_flag)
        checkbox_entry.place(x=5, y=100)

        # Flag for Camera and Verification options
        self.gaze_calib_flag = tk.BooleanVar(master=self.content_frame, value=True)
        checkbox_entry = tk.Checkbutton(self.content_frame, text="Gaze Calibration", variable=self.gaze_calib_flag)
        checkbox_entry.place(x=5, y=148)

        self.ci_calib_flag = tk.BooleanVar(master=self.content_frame, value=False)
        checkbox_entry = tk.Checkbutton(self.content_frame, text="Auto CI Calibration", variable=self.ci_calib_flag)
        checkbox_entry.place(x=140, y=148)

        # Configure CI and DI thresholds
        self.ci_threshold_entry = tk.Entry(self.content_frame)
        self.ci_threshold_entry.insert(0, 'CI threshold')
        self.ci_threshold_entry.place(x=150, y=175, width=100)

        self.di_threshold_entry = tk.Entry(self.content_frame)
        self.di_threshold_entry.insert(0, 'DI threshold')
        self.di_threshold_entry.place(x=150, y=200, width=100)

        self.ci_calib_flag.trace_add(
            'write',
            lambda *args: manual_ci_threshold_visible(
                self.ci_calib_flag,
                [self.ci_threshold_entry, self.di_threshold_entry],
                [{'x': 150, 'y': 175, 'width': 100}, {'x': 150, 'y': 200, 'width': 100}],
                * args
            )
        )

        # Configure RIB decision window size and proportion
        self.decision_window_size_entry = tk.Entry(self.content_frame)
        self.decision_window_size_entry.insert(0, 'window size [frames]')
        self.decision_window_size_entry.place(x=5, y=245, width=130)

        self.proportion_entry = tk.Entry(self.content_frame)
        self.proportion_entry.insert(0, 'threshold [%]')
        self.proportion_entry.place(x=150, y=245, width=100)

        self.menu_frame.bind_all(
            "<Button-1>", lambda event: restore_placeholder(
                event,
                [self.ci_threshold_entry, self.di_threshold_entry],
                ['CI threshold', 'DI threshold']
            )
        )

    def open_calib_screen(self):
        self.master.withdraw()  # Hide the main screen
        calib_screen(self.master, self.cima)

    def open_csv_analyzer_screen(self):
        csv_analyzer_screen(self.master, self.cima)

    def open_video_analyzer_screen(self):
        self.master.withdraw()  # Hide the main screen
        video_analyzer_screen(self.master, self.cima, self.visualize_flag.get())

    def _get_threshold_entries(self):
        # set new manual thresholds for CI and DI:
        try:
            ci_threshold = self.ci_threshold_entry.get()
            ci_threshold = float(ci_threshold)
            config.analyze.CI_THRESHOLD = ci_threshold
            print(f"Got new constant CI threshold: {ci_threshold}")
        except Exception:
            print(f"Using last CI threshold: {config.analyze.CI_THRESHOLD}")

        try:
            di_threshold = self.di_threshold_entry.get()
            di_threshold = float(di_threshold)
            config.analyze.DI_THRESHOLD = di_threshold
            print(f"Got new constant DI threshold: {di_threshold}")
        except Exception:
            print(f"Using last DI threshold: {config.analyze.DI_THRESHOLD}")

        try:
            window_size = self.decision_window_size_entry.get()
            window_size = int(window_size)
            if window_size > 0:
                config.analyze.DECISION_WINDOW_SIZE = window_size
        except Exception:
            pass

        try:
            proportion = self.proportion_entry.get()
            proportion = float(proportion)
            if 0.0 < proportion < 1.0:
                config.analyze.DECISION_THRESHOLD_FOR_ALERT = proportion
        except Exception:
            pass

    def __auto_ci_calib_process(self):
        screen_number = get_laptop_monitor_number()

        threshold_valid_flag = False
        abort_flag = False
        focal_area_ci_threshold = []

        while not threshold_valid_flag and not abort_flag:
            # Calibrate CI thresholds for each focal area
            ci_usercalib = CI_UserCalib(self.cima, screen_number=screen_number)
            calib_success = ci_usercalib.run(calib=self.gaze_calib_flag.get())
            focal_area_ci_threshold = []
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
                    focal_area_ci_threshold.append(
                        np.round(max(config.analyze.MIN_CI_THRESHOLD, (p_90 * 1.1)), 1)
                    )
                    print(f"\tthrehold: {focal_area_ci_threshold[key]}")

                if all(x <= config.analyze.MAX_CI_THRESHOLD for x in focal_area_ci_threshold):
                    threshold_valid_flag = True
            else:
                abort_flag = True

        if not abort_flag and threshold_valid_flag:
            return focal_area_ci_threshold
        else:
            return None

    def __manual_ci_calib(self):
        calib_success = True
        self._get_threshold_entries()

        # Check for Gaze pre-calibration
        if self.gaze_calib_flag.get():
            user_calib = UserCalib(self.cima)
            screen_number = get_laptop_monitor_number()
            calib_success = user_calib.run(screen_number)
        return calib_success

    def open_camera(self):
        self.master.withdraw()  # Hide the main screen

        # Check for automatic / manual thresholds
        if self.ci_calib_flag.get():
            focal_area_ci_threshold = self.__auto_ci_calib_process()
            if focal_area_ci_threshold is not None:
                self.cima.analyze(debug=True, save_video=False, live_stat=False, target_fps=config.analyze.TARGET_FPS,
                                  focal_area_ci_threshold=focal_area_ci_threshold)

        elif self.__manual_ci_calib():
            self.cima.analyze(debug=True, save_video=False, live_stat=False, target_fps=config.analyze.TARGET_FPS)

        self.master.deiconify()  # Show the screen again

    def open_verification(self):
        self._get_threshold_entries()

        self.master.withdraw()  # Hide the main screen
        verification_screen(self.master, self.cima)

    def run_in_background(self):
        self._get_threshold_entries()

        self.master.withdraw()  # Hide the main screen

        focal_area_ci_threshold = self.__auto_ci_calib_process()
        if focal_area_ci_threshold is not None:
            if self.cima.run(thresholds=focal_area_ci_threshold, target_fps=config.analyze.TARGET_FPS):
                messagebox.showinfo("CI detect", "Please rest from the screen due to CI detection")

        self.master.deiconify()  # Show the screen again

    def log_out(self):
        if messagebox.askokcancel("Log Out", "Are you sure you want to log out?"):
            self.master.destroy()
            self.login_screen.master.deiconify()  # Show the login screen again
            self.login_screen.id_entry.delete(0, tk.END)  # Clear the user ID entry
