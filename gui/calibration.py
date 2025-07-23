import random
import tkinter as tk
from tkinterdnd2 import TkinterDnD

from config import Config
from helper.file import *
from helper.monitor import get_laptop_monitor_number
from gui.calib_validation_screen import calib_validation_screen
from gui.cameraCalibScreen import cameraCalibScreen
from gui.movingDotScreen import movingDotScreen
from gui.CI_Screen import CI_Screen
from cima.cima import CIMA


def restore_placeholder(event, entries, placeholder_texts):
    for entry, placeholder_text in zip(entries, placeholder_texts):
        if event.widget != entry and entry.get() == "":
            entry.insert(0, placeholder_text)


class calib_screen:
    def __init__(self, prev_screen, cima: CIMA):
        self.random_amount_entry = None
        self.distance_entry = None
        self.pos_indx_entry = None
        self.content_frame = None
        self.menu_frame = None
        self.visualize_flag = None
        self.reset_calib_flag = None
        self.random_pattern_flag = None

        self.prev_screen = prev_screen
        self.master = TkinterDnD.Tk()
        self.cima = cima

        self.master.title(Config.APP_TITLE)
        self.center_window("calibration")
        self.setup_ui()

        # Set up the window close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Get laptop monitor number:
        self.screen_number = get_laptop_monitor_number()

        # test configurations:
        self.mv_dot_positions = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        self.ci_positions = [
            (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
            (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
            (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)
        ]

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
        tk.Button(self.menu_frame, text="camera calib", command=self.open_camera_calib_screen).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="moving dot", command=self.open_moving_dot_screen).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="Convergence", command=self.open_convergence_screen).pack(fill=tk.X, pady=10)
        tk.Button(self.menu_frame, text="Divergence", command=self.open_divergence_screen).pack(fill=tk.X, pady=10)

        if not Config.IS_PRODUCTION:
            tk.Button(self.menu_frame, text="validate", command=self.open_validate_screen).pack(fill=tk.X, pady=10)
            tk.Button(self.menu_frame, text="one dot", command=self.open_one_dot_tracking_screen).pack(fill=tk.X, pady=10)
            self.random_pattern_flag = tk.BooleanVar(master=self.menu_frame)
            tk.Checkbutton(self.menu_frame, text="Random pattern", variable=self.random_pattern_flag).pack(fill=tk.X, pady=10)

        self.visualize_flag = tk.BooleanVar(master=self.menu_frame, value=True)
        self.reset_calib_flag = tk.BooleanVar(master=self.menu_frame, value=True)
        tk.Checkbutton(self.menu_frame, text="visualize", variable=self.visualize_flag).pack(fill=tk.X, pady=10)
        tk.Checkbutton(self.menu_frame, text="reset calib", variable=self.reset_calib_flag).pack(fill=tk.X, pady=10)

        if not Config.IS_PRODUCTION:
            tk.Button(self.menu_frame, text="train Gaze model", command=self.train_gaze_model).pack(fill=tk.X, pady=10,
                                                                                                    side=tk.BOTTOM)
            tk.Button(self.menu_frame, text="train CI-model", command=self.train_ci_model).pack(fill=tk.X, pady=10,
                                                                                                side=tk.BOTTOM)

        # Add spacer
        tk.Frame(self.menu_frame, height=20, bg='lightgray').pack(fill=tk.X, expand=True)

        # Create a frame for the content
        self.content_frame = tk.Frame(self.master)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Welcome message
        tk.Label(self.content_frame, text=f"Welcome, User {self.cima.user_id}", font=("Arial", 24)).pack(pady=20)

        if not Config.IS_PRODUCTION:
            # Position index for "one dot" screen
            self.pos_indx_entry = tk.Entry(self.content_frame)
            self.pos_indx_entry.insert(0, "pos index")
            self.pos_indx_entry.place(x=5, y=242, width=70)

            # Distance between 2 dots for "one dot" screen
            self.distance_entry = tk.Entry(self.content_frame)
            self.distance_entry.insert(0, "distance")
            self.distance_entry.place(x=90, y=242, width=70)

            # Amount of random dots for "one dot" screen
            self.random_amount_entry = tk.Entry(self.content_frame)
            self.random_amount_entry.insert(0, "random_amount")
            self.random_amount_entry.place(x=5, y=290, width=100)
            self.menu_frame.bind_all(
                "<Button-1>",
                lambda event: restore_placeholder(
                    event,
                    [self.pos_indx_entry, self.distance_entry, self.random_amount_entry],
                    ['pos index', 'distance', 'random amount']
                )
            )

    def train_gaze_model(self):
        self.cima.eyes_analyzer.gaze_model_train()

    def train_ci_model(self):
        self.cima.eyes_analyzer.ci_model_train()

    def open_camera_calib_screen(self):
        self.master.withdraw()  # Hide the main screen
        calib = cameraCalibScreen()
        calib.run(self.screen_number, Config.CAMERA_CALIB_FILE_PATH)
        self.master.deiconify()  # Show the prev screen again

    def open_moving_dot_screen(self):
        # moving dot test plan
        self.master.withdraw()  # Hide the main screen
        test = movingDotScreen(self.cima, self.mv_dot_positions)
        if test.run(self.screen_number, calib=self.reset_calib_flag.get()):
            self.cima.analyze(test.get_video_filename(), view_analyze=self.visualize_flag.get())
        test.close()

        self.master.deiconify()  # Show the prev screen again

    def __get_pos_indx_entry(self):
        try:
            pos_indx = int(self.pos_indx_entry.get())
        except Exception:
            pos_indx = 0

        if pos_indx < 0:
            pos_indx = 0
        elif pos_indx > len(self.mv_dot_positions) - 1:
            pos_indx = len(self.mv_dot_positions) - 1

        return pos_indx

    def __get_distance_entry(self):
        try:
            distance = int(self.distance_entry.get())
        except Exception:
            distance = 0

        if distance < 0:
            distance = 0

        return distance

    def __get_random_pairs(self):
        try:
            random_amound = int(self.random_amount_entry.get())
        except Exception:
            random_amound = 0

        if random_amound < 0:
            random_amound = 0

        return [(round(random.uniform(0.1, 0.9), 2), round(random.uniform(0.1, 0.9), 2)) for _ in range(random_amound)]

    def open_one_dot_tracking_screen(self):
        pos_indx = self.__get_pos_indx_entry()
        distance = self.__get_distance_entry()
        random_flag = self.random_pattern_flag.get()

        if distance == 0:
            test_name = f'one_dot_test_pos_random' if random_flag else f'one_dot_test_pos{pos_indx}'
            positions = self.__get_random_pairs() if random_flag else [self.mv_dot_positions[pos_indx]]

            test = movingDotScreen(self.cima, positions, test_name)
        elif distance > 0:
            test_name = f'one_dot_convergence_pos_random' if random_flag else f'one_dot_convergence_pos{pos_indx}'
            positions = self.__get_random_pairs() if random_flag else [self.ci_positions[pos_indx]]

            test = CI_Screen(test_name, np.array([0, distance]), positions, self.cima)
        else:
            test_name = f'one_dot_divergence_pos_random' if random_flag else f'one_dot_divergence_pos{pos_indx}'
            positions = self.__get_random_pairs() if random_flag else [self.ci_positions[pos_indx]]
            test = CI_Screen(test_name, np.array([0, distance]), positions, self.cima)

        self.master.withdraw()  # Hide the main screen
        if test.run(self.screen_number, calib=self.reset_calib_flag.get()):
            self.cima.analyze(test.get_video_filename(), view_analyze=self.visualize_flag.get())
        test.close()

        self.master.deiconify()  # Show the prev screen again

    def open_convergence_screen(self):
        # Convergence test plan
        distance_mm = np.arange(30, 100, 20)  # unit [mm]
        distance_mm = np.insert(distance_mm, 0, 0)

        self.master.withdraw()  # Hide the main screen
        test = CI_Screen("convergence", distance_mm, self.ci_positions, self.cima)
        if test.run(self.screen_number, calib=self.reset_calib_flag.get()):
            self.cima.analyze(test.get_video_filename(), view_analyze=self.visualize_flag.get())
        test.close()
        self.master.deiconify()  # Show the prev screen again

    def open_divergence_screen(self):
        # Divergence test plan
        distance_mm = np.arange(30, 50, 5)  # unit [mm]
        distance_mm = np.insert(distance_mm, 0, 0)

        self.master.withdraw()  # Hide the main screen
        test = CI_Screen("divergence", distance_mm, self.ci_positions, self.cima)
        if test.run(self.screen_number, calib=self.reset_calib_flag.get()):
            self.cima.analyze(test.get_video_filename(), view_analyze=self.visualize_flag.get())
        test.close()
        self.master.deiconify()  # Show the prev screen again

    def open_validate_screen(self):
        self.master.withdraw()  # Hide the main screen
        calib_validation_screen(self.cima, self)
