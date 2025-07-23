import cv2
from tkinter import filedialog, messagebox

from helper.file import *
from helper.helper import *
from helper.monitor import *
from cima.cima import CIMA


class calib_validation_screen:
    def __init__(self, cima: CIMA, prev_screen):
        self.cima = cima
        self.prev_screen = prev_screen

        self.setup_ui()

    def setup_ui(self):
        # select mp4 file
        filetypes = (('MP4 files', '*.mp4'), ('All files', '*.*'))
        filename = filedialog.askopenfilename(title='Select an MP4 File', initialdir='captures', filetypes=filetypes)
        if filename:
            test = validate_calib(self.cima, filename)
            test.run()

        self.on_closing()

    def on_closing(self):
        self.prev_screen.master.deiconify()  # Show the prev screen again


class validate_calib:
    def __init__(self, cima: CIMA, video_path):
        self.cima = cima

        if not video_path.lower().endswith('.mp4'):
            messagebox.showerror("Error", "Please drop a MP4 file.")
            return

        self.video_path = video_path
        base_name = os.path.splitext(video_path)[0]
        self.csv_file = base_name + ".csv"

        # Get test name
        parts = base_name.split("_")
        test_name = "_".join(parts[-2:-1])

        # Get test info from CSV file
        self.dots_col = []
        if test_name == "moving-dot":
            self.dots_col.append((2, 3))
        elif test_name == "convergence":
            self.dots_col.append((3, 4))
            self.dots_col.append((5, 6))

        with open(self.csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            self.header = next(reader)  # Read the header
            self.data = list(reader)  # Read the rest of the data
            file.close()

        # Dot properties
        self.radius_focus = 8
        self.radius = 25

        self.test_name = "validation"
        self.state = TEST_IDLE
        self.center = None
        self.screen_width = None
        self.screen_height = None
        self.frame = None

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

    def clear_screen(self):
        self.frame[:] = CV2_COLOR_BLACK

    def draw_circle(self, color):
        cv2.circle(self.frame, self.center, self.radius, color, -1)
        cv2.circle(self.frame, self.center, self.radius_focus, CV2_COLOR_BLACK, -1)

    def update_color(self):
        if self.state == TEST_IDLE:
            return CV2_COLOR_WHITE
        elif self.state == TEST_START:
            return CV2_COLOR_RED
        else:
            return None

    def run(self):
        self.create_fullscreen_window()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        def_delay = 1000 / fps
        delay = def_delay
        cur_frame = 0

        while True:
            success, webcam_frame = cap.read()
            if not success:
                break

            #  Draw the circles
            self.clear_screen()
            for i, (xcol, ycol) in enumerate(self.dots_col):
                # get new dot location
                self.center = (int(self.data[cur_frame][xcol]), int(self.data[cur_frame][ycol]))
                self.draw_circle(CV2_COLOR_WHITE)

            # show image
            cv2.imshow(self.test_name + "-video", webcam_frame)

            # Update test's frame
            cv2.imshow(self.test_name, self.frame)

            key = cv2.waitKey(int(delay)) & 0xFF
            if key == ord('a'):
                # go to prev frame
                cur_frame = max(0, cur_frame - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            elif key == ord('d'):
                # go to next frame
                cur_frame = min(n_frames - 1, cur_frame + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            elif key == ord('w'):
                # faster speed
                delay = max(def_delay / 16, delay / 2)
            elif key == ord('s'):
                # slower speed
                delay = min(def_delay * 16, delay * 2)
            elif key == ord('f'):
                # freeze
                delay = 0
            elif key == ord('c'):
                # stop freeze
                delay = def_delay
            elif key == 27:
                # handle APP exit
                break

            # Update the current frame number
            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        cap.release()
        cv2.destroyAllWindows()
