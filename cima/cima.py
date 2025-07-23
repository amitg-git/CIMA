import cv2
import logging
import time
import numpy as np

import config
from helper.file import *
import cima.analyzer as analyzer
from helper.monitor import get_laptop_monitor_number, monitor, performance
from helper.statistics import CI_RealTimeDetect
from helper.helper import *


def draw_all_landmarks_on_face(frame, mp_results):
    if mp_results.multi_face_landmarks:
        for face_landmarks in mp_results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Draw a small circle for each landmark
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Optionally, add the landmark index as text
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


def draw_facemesh_on_face(frame, mp_results, facemesh):
    if mp_results.multi_face_landmarks:
        face_landmarks = mp_results.multi_face_landmarks[0]
        for i, pair in enumerate(facemesh):
            p = pair[0]
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            x, y = int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h)

            # Draw a small circle for each landmark
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Optionally, add the landmark index as text
            cv2.putText(frame, str(p), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def draw_facemesh_point_on_face(frame, mp_results, points):
    if mp_results.multi_face_landmarks:
        for p in points:
            face_landmarks = mp_results.multi_face_landmarks[0]
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            x, y = int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h)

            # Draw a small circle for each landmark
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Optionally, add the landmark index as text
            cv2.putText(frame, str(p), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


class CIMA:
    def __init__(self, user_id=None):
        self.user_id = user_id

        # Create a logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

        screen_num = get_laptop_monitor_number()
        self.scr = monitor(screen_num).info
        self.pause = False

        # Optional files
        self.window_name: str = config.Config.APP_TITLE
        self.video_name = None
        self.save_video = False
        self.video_writer = None
        self.eyes_analyzer = analyzer.eyes_analyzer()
        self.analyzer_logger = analyzer.analyzer_logger()

        self.grid_win_name = "Grid Visualization"

    def reset(self):
        self.eyes_analyzer.reset()

    def init_output_file(self, is_live: bool):
        # get the csv file name or create a new one
        if is_live:
            csv_file_name = get_unique_filename(f"Statistics/{self.user_id}_live_1.csv")
        else:
            csv_file_name = os.path.splitext(self.video_name)[0] + ".csv"

        # initialize the logger
        self.analyzer_logger.init_csv_output(csv_file_name)

    def video_save_init(self, frame_width, frame_height, fps):
        if self.save_video and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            name = self.video_name if self.video_name is not None else 'live'
            cap_name = get_unique_filename(f"captures/live/{self.user_id}_{name}_1.mp4")
            self.video_writer = cv2.VideoWriter(cap_name, fourcc, fps, (frame_width, frame_height))

    def video_write(self, frame):
        if self.save_video:
            self.video_writer.write(frame)

    def video_release(self):
        if self.save_video:
            self.video_writer.release()
            self.video_writer = None

    def __create_grid_window(self):
        cv2.namedWindow(self.grid_win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.grid_win_name, self.scr.x, self.scr.y)
        cv2.setWindowProperty(self.grid_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show_eyes_landmarks(self, frame):
        draw_facemesh_on_face(frame, self.eyes_analyzer.mp_results, self.eyes_analyzer.mp_face_mesh.FACEMESH_RIGHT_EYE)
        draw_facemesh_on_face(frame, self.eyes_analyzer.mp_results, self.eyes_analyzer.mp_face_mesh.FACEMESH_LEFT_EYE)

    def show_ear_param_landmarks(self, frame):
        draw_facemesh_on_face(frame, self.eyes_analyzer.mp_results, config.analyze.FACEMESH_EAR_LEFT_EYE)
        draw_facemesh_on_face(frame, self.eyes_analyzer.mp_results, config.analyze.FACEMESH_EAR_RIGHT_EYE)

    def analyze(self, video_name: str = None,
                debug: bool = False,
                save_video: bool = False,
                live_stat: bool = False,
                view_analyze: bool = False,
                focal_area_ci_threshold=None,
                target_fps: float = -1.0):
        self.video_name = video_name
        self.save_video = save_video

        self.eyes_analyzer.reset_internal_data()

        # Open video capture for camera or mp4 file
        if video_name is None:
            cap = cv2.VideoCapture(0)
            self.logger.debug("Analyzing Live video")
            is_live = True
        else:
            cap = cv2.VideoCapture(video_name)
            number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frame = 0
            process_percentage_to_print = 10
            self.logger.debug(f"Analyzing file: {video_name}")
            is_live = False

        if not is_live or live_stat:
            self.init_output_file(is_live)

        # get capture information
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # init save instance if needed
        self.video_save_init(frame_width, frame_height, fps)

        manual_exit_flag = False
        start_time = time.time()

        if debug and is_live:
            self.__create_grid_window()

        fps = performance()
        while cap.isOpened():
            if not self.pause:
                # read one frame from camera
                success, frame = cap.read()
                if not success:
                    if is_live:
                        self.logger.warning("Ignoring empty camera frame.")
                        continue
                    else:
                        break

                # analyze frame to get eyes information
                self.eyes_analyzer.analyze_frame(frame, target_fps, focal_area_ci_threshold=focal_area_ci_threshold)

                # print for each eye the mediaPipe landmarks
                # self.show_ear_param_landmarks(frame)

                # draw_facemesh_point_on_face(frame, self.eyes_analyzer.mp_results, np.arange(468, 478))
                # self.show_eyes_landmarks(frame)
                # draw_all_landmarks_on_face(frame, self.eyes_analyzer.mp_results)

                if debug or not is_live:
                    self.eyes_analyzer.add_info_to_frame(frame, is_live, fps=fps.get_fps())

                if not is_live or live_stat:
                    self.analyzer_logger.update(self.eyes_analyzer.get_data())
                    self.analyzer_logger.stat_push()

                if is_live or view_analyze:
                    cv2.imshow(self.window_name, frame)
                self.video_write(frame)

                if debug and is_live:
                    # view focal point in low resolution
                    vis_frame = self.eyes_analyzer.visualize_fcal_point(640, 480)
                    cv2.imshow(self.grid_win_name, vis_frame)

                if not is_live:
                    processed_frame += 1
                    process_percentage = round((processed_frame / number_of_frames) * 100)
                    if process_percentage >= process_percentage_to_print:
                        process_percentage_to_print += 10
                        print(f'process {process_percentage}[%], {round(time.time() - start_time, 2)}[sec]')

            if is_live or view_analyze:
                # user pressed Esc in keyboard or user close the window
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # pause the analysis
                    self.pause = not self.pause
                elif key == ord('r'):  # reset the calibration
                    self.reset()
                elif key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    manual_exit_flag = True
                    break

        print(f"process time:{round(time.time() - start_time, 2)}[sec]")
        if is_live or not manual_exit_flag:
            self.analyzer_logger.close()
        self.video_release()

        cap.release()
        cv2.destroyAllWindows()

        return manual_exit_flag

    def __create_exit_window(self):
        window_width, window_height = 300, 200

        top_middle_x = self.scr.x + (self.scr.width // 2) - (window_width // 2)
        top_middle_y = self.scr.y

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        cv2.moveWindow(self.window_name, top_middle_x, top_middle_y)

        frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Add instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = CV2_COLOR_WHITE
        thickness = 2

        text1 = "Press 'q' to quit"

        # Calculate text size and position for centering
        (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)

        # Position text1 near the top center
        text1_x = (window_width - text1_width) // 2
        text1_y = (window_height // 2) - 20

        cv2.putText(frame, text1, (text1_x, text1_y), font, font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow(self.window_name, frame)

    def run(self, thresholds, target_fps: float = -1.0):
        today = time.strftime('%Y%m%d')
        # Start Real-time Analysis

        ci_detector = CI_RealTimeDetect(
            window_size=config.analyze.DECISION_WINDOW_SIZE,
            thresholds=thresholds,
            threshold_for_alert=config.analyze.DECISION_THRESHOLD_FOR_ALERT,
            save_path=get_unique_filename(f"user_reports/{self.user_id}_{today}_rib_1.csv")
        )
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Failed to open camera")
            return False

        self.__create_exit_window()

        ci_alert = False
        while not ci_alert:
            success, frame = cap.read()
            if not success:
                continue

            # analyze frame to get eyes information
            is_face_detect, is_eyes_open, is_gaze_valid = self.eyes_analyzer.analyze_frame(frame, target_fps)
            if is_gaze_valid:
                # add CI distance to classifier
                ci_dist = self.eyes_analyzer.ci_dist.get_last()
                focal_area = self.eyes_analyzer.cur_focus_cell.get()
                ci_detector.add(focal_area, ci_dist)
                if ci_detector.get():
                    ci_alert = True

            # handle APP exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        # Clean and exit
        ci_detector.save_stat()
        cap.release()
        cv2.destroyAllWindows()
        return ci_alert
