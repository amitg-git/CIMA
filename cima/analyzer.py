import cv2
import time
import mediapipe as mp

import gaze
import helper.monitor
from helper.monitor import get_focal_area_index, get_focal_area_pos
from helper.file import logvar, csv_statistics
from helper.statistics import visualize_grid_data, ClassificationAverage, AverageSimple, MovingWindow
import config

from LaserGaze.GazeProcessor import GazeProcessor
from LaserGaze.VisualizationOptions import VisualizationOptions
from cima.focal_estimation import *
from cima.convergence_model import *
from cima.DataCollection import DataCollection


class eyes_analyzer:
    # C:\Users\<username>\AppData\Local\Programs\Python\Python39\Lib\site-packages\mediapipe\python\solutions\face_mesh_connections.py
    # C:\Users\<username>\AppData\Local\Programs\Python\Python39\Lib\site-packages\mediapipe\python\solutions\face_mesh_test.py
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_results = None

        # Initialize internal data
        self.to_show_eyes = False
        self.to_show_gazes = True

        # Initialize things for MediaPipe
        vo = VisualizationOptions()
        self.gp = GazeProcessor(visualization_options=vo)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize internal data
        monitor = helper.monitor.monitor()
        self.scr = monitor.info
        self.vis_focus_grid_rows = self.vis_focus_grid_cols = 3
        self.screen_grid = (self.scr.width, self.scr.height, self.vis_focus_grid_rows, self.vis_focus_grid_cols)

        self.fps_timetag = 0
        self.ear_left = 1.0  # Eye Aspect Ratio
        self.ear_right = 1.0  # Eye Aspect Ratio
        self.eyes_open = True
        self.blink_debouncer_begin = 0
        self.ci_predicted_label_name = 'NA'
        self.avg_dgaze = AverageSimple(window_size=5)
        self.cur_focus_cell = ClassificationAverage(window_size=config.analyze.FOCUS_CELL_AVG_SIZE)
        # self.ci_status_code = ClassificationAverage(window_size=config.analyze.CI_STATUS_AVG_SIZE)
        self.ci_dist = MovingWindow(config.analyze.CI_DIST_WINDOW_SIZE, func=np.median)

        self.gaze_predictor: FocalEstimationModel = FocalEstimationModel()
        self.ci_predictor: ConvergenceModel = ConvergenceModel()
        self.models_init()

        self.dc = DataCollection(gaze_model_features_path=self.gaze_predictor.features_path,
                                 ci_model_features_path=self.ci_predictor.features_path)
        self.datadict: dict = {}

        try:
            calib_data = joblib.load(config.Config.CAMERA_CALIB_FILE_PATH)
            self.normalized_focal_x = calib_data['normalized_fx']
        except Exception as e:
            print(f"Warning: Failed to get normalized_fx, use default value. {e}")
            self.normalized_focal_x = 1.05

    def reset_internal_data(self):
        self.fps_timetag = 0
        self.ear_left = 1.0  # Eye Aspect Ratio
        self.ear_right = 1.0  # Eye Aspect Ratio
        self.eyes_open = True
        self.blink_debouncer_begin = 0
        self.ci_predicted_label_name = 'NA'
        self.avg_dgaze = AverageSimple(window_size=5)
        self.cur_focus_cell = ClassificationAverage(window_size=config.analyze.FOCUS_CELL_AVG_SIZE)
        # self.ci_status_code = ClassificationAverage(window_size=config.analyze.CI_STATUS_AVG_SIZE)
        self.ci_dist = MovingWindow(5, func=np.median)

        self.dc = DataCollection(gaze_model_features_path=self.gaze_predictor.features_path,
                                 ci_model_features_path=self.ci_predictor.features_path)

        self.datadict = {
            'depth_left_cm': None,
            'depth_right_cm': None,
            'scr_dist': None,
            'head_pose_xy': None,
            'left_pupil_scr_xy': None,
            'left_proj_point_xyz': None,
            'left_proj_point_scr_xy': None,
            'left_gaze_vector_xyz': None,
            'right_pupil_xyz': None,
            'right_pupil_scr_xy': None,
            'right_proj_point_xyz': None,
            'right_proj_point_scr_xy': None,
            'right_gaze_vector_xyz': None,
            'dGaze_xy': None
        }

    def reset(self):
        self.reset_internal_data()
        self.gp.reset()

    def get_data(self):
        return self.dc

    def analyze_frame(self, frame, target_fps=-1, *, focal_area_ci_threshold=None):
        """
        Analyze one frame of data.

        Input:
            frame - capture frame
            target_fps_ms - target FPS in sec. -1 for unlimited
        Output:
            tuple if face has been detected and eyes have been localized
        """
        # Make fps as we want
        if target_fps != -1:
            cur_time = time.time()
            dt = cur_time - self.fps_timetag
            if dt < target_fps:
                rem = target_fps - dt
                time.sleep(rem)
            self.fps_timetag = cur_time

        # Start analyze
        frame.flags.writeable = False  # improve performance
        self.mp_results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True

        self.__estimate_ear(frame)

        is_face_detect = False if not self.mp_results.multi_face_landmarks else True
        is_eyes_open = self.eyes_open

        is_gaze_valid = False
        if self.eyes_open:
            blink_debouncer = time.time() - self.blink_debouncer_begin
            if blink_debouncer > config.analyze.BLINK_DEBOUNCER_SEC:
                self.__estimate_eyes_depth(frame)
                is_gaze_valid = self.__estimate_gaze_directions(frame)

                # update calculated data
                if is_gaze_valid:
                    self.dc.update(self.__split_multi_coord_dict())

                    # Estimate CI and Gaze Focal at screen plane
                    focal_area_index, predicted_coords = self.__focal_area_predict()
                    self.cur_focus_cell.add(focal_area_index)
                    self.__estimate_ci(focal_area_ci_threshold)
        else:
            self.blink_debouncer_begin = time.time()

        return is_face_detect, is_eyes_open, is_gaze_valid

    def __split_multi_coord_dict(self):
        new_datadict = {}
        for key in self.datadict.keys():
            if key.endswith('_xy'):
                new_key_x = key.replace('_xy', '_x')
                new_key_y = key.replace('_xy', '_y')
                new_datadict[new_key_x] = self.datadict[key][0]
                new_datadict[new_key_y] = self.datadict[key][1]
            elif key.endswith('_xyz'):
                new_key_x = key.replace('_xyz', '_x')
                new_key_y = key.replace('_xyz', '_y')
                new_key_z = key.replace('_xyz', '_z')
                new_datadict[new_key_x] = self.datadict[key][0]
                new_datadict[new_key_y] = self.datadict[key][1]
                new_datadict[new_key_z] = self.datadict[key][2]
            else:
                new_datadict[key] = self.datadict[key]

        return new_datadict

    def __estimate_ci(self, focal_area_ci_threshold=None):
        self.avg_dgaze.add(self.dc.get('dGaze_x'))

        input_features = self.dc.get_ci_model_features()
        ci_dist = self.ci_predictor.predict(input_features, has_units=False)
        # print(f"ci_dist={ci_dist}")
        self.ci_dist.add(np.clip(ci_dist, 0, None))
        dist = self.ci_dist.get()

        if focal_area_ci_threshold is None:
            # Constant thresholds
            if dist > config.analyze.CI_THRESHOLD:
                self.ci_predicted_label_name = "CI"
            elif dist < config.analyze.DI_THRESHOLD:
                self.ci_predicted_label_name = "DI"
            else:
                self.ci_predicted_label_name = "NO"
        else:
            #  Per focal area threshold (only CI/NO)
            focal_area = self.cur_focus_cell.get()
            if dist > focal_area_ci_threshold[focal_area]:
                self.ci_predicted_label_name = "CI"
            else:
                self.ci_predicted_label_name = "NO"

    def __estimate_ear(self, frame):
        self.ear_left = gaze.calc_ear(frame, self.mp_results, config.analyze.FACEMESH_EAR_LEFT_EYE)
        self.ear_right = gaze.calc_ear(frame, self.mp_results, config.analyze.FACEMESH_EAR_RIGHT_EYE)
        self.eyes_open = self.is_eyes_open()

    def is_eyes_open(self):
        if self.ear_left < config.analyze.EAR_OPEN_THRESHOLD:
            return False

        if self.ear_right < config.analyze.EAR_OPEN_THRESHOLD:
            return False

        return True

    def __estimate_eyes_depth(self, frame):
        # Get Left and Right iris depth from the image
        depth_left_cm = gaze.calc_iris_depth(frame, self.mp_results, self.mp_face_mesh.FACEMESH_LEFT_IRIS,
                                             normalized_focal_x=self.normalized_focal_x)
        depth_right_cm = gaze.calc_iris_depth(frame, self.mp_results, self.mp_face_mesh.FACEMESH_RIGHT_IRIS,
                                              normalized_focal_x=self.normalized_focal_x)
        screen_distance_cm = (depth_left_cm + depth_right_cm) / 2.0

        self.datadict['depth_left_cm'] = depth_left_cm
        self.datadict['depth_right_cm'] = depth_right_cm
        self.datadict['scr_dist'] = screen_distance_cm

    def check_for_sufficient_gaze_detection(self, frame):
        param = self.gp.update(frame)
        return param['valid']

    def __estimate_gaze_directions(self, frame):
        head_pose = gaze.gaze(frame, self.mp_results)
        param = self.gp.update(frame)

        if head_pose is None or not param['valid']:
            return False

        self.datadict['head_pose_xy'] = np.array(head_pose)
        self.datadict['left_pupil_xyz'] = param['left_pupil_xyz']
        self.datadict['left_pupil_scr_xy'] = param['left_pupil_scr_xy']
        self.datadict['left_proj_point_xyz'] = param['left_proj_point_xyz']
        self.datadict['left_proj_point_scr_xy'] = param['left_proj_point_scr_xy']
        self.datadict['left_gaze_vector_xyz'] = param['left_gaze_vector_xyz']

        self.datadict['right_pupil_xyz'] = param['right_pupil_xyz']
        self.datadict['right_pupil_scr_xy'] = param['right_pupil_scr_xy']
        self.datadict['right_proj_point_xyz'] = param['right_proj_point_xyz']
        self.datadict['right_proj_point_scr_xy'] = param['right_proj_point_scr_xy']
        self.datadict['right_gaze_vector_xyz'] = param['right_gaze_vector_xyz']

        dGaze = abs(param['left_proj_point_scr_xy'] - param['right_proj_point_scr_xy'])
        self.datadict['dGaze_xy'] = dGaze[:2]  # only (x,y)

        return True

    def add_info_to_frame(self, frame, is_live: bool = False, fps: float = 0.0):
        # Image shape
        height, width, _ = frame.shape

        cv2.putText(frame, f"FPS:{fps:04.1f}",
                    (int(width * 0.01), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Dist:{self.dc.get('scr_dist'):.1f}[cm]",
                    (int(width * 0.65), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_live:
            cv2.putText(frame, f"dGazeX:{self.avg_dgaze.get():05.1f}[{self.ci_predicted_label_name}]",
                        (int(width * 0.54), 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"dGazeX:{self.dc.get('dGaze_x'):05.1f}[{self.ci_predicted_label_name}]",
                        (int(width * 0.54), 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        eyes_status = "open" if self.is_eyes_open() else "close"
        cv2.putText(frame, f"eyes:{eyes_status}",
                    (int(width * 0.72), 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_live:
            cv2.putText(frame, f"SPCD:{int(np.round(self.ci_dist.get()))}[mm]",
                        (int(width * 0.65), 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def visualize_fcal_point(self, win_width, win_height):
        return visualize_grid_data(win_width, win_height,
                                   self.vis_focus_grid_rows, self.vis_focus_grid_cols,
                                   self.cur_focus_cell.get(),
                                   self.ci_predicted_label_name)

    def gaze_model_train(self):
        focal_model_train()
        self.models_init()

    def ci_model_train(self):
        ci_model_train()
        self.models_init()

    def models_init(self):
        self.gaze_predictor = FocalEstimationModel(
            model_dir=GAZE_MODEL_SAVE_DIR
        )
        self.ci_predictor = ConvergenceModel(  # Re-init for loading
            model_dir=CI_MODEL_SAVE_DIR
        )

        self.gaze_predictor.load()
        self.ci_predictor.load()

    def __focal_area_predict(self):
        input_features = self.dc.get_gaze_model_features()
        predicted_coords = self.gaze_predictor.predict(input_features)[0]

        col_index, row_index = get_focal_area_pos(predicted_coords[0], predicted_coords[1], *self.screen_grid)
        focal_area_index = (row_index * self.vis_focus_grid_cols) + col_index

        # self.dc.update_focal_xy(predicted_coords[0], predicted_coords[1])
        self.dc.update_focal_area(col_index, row_index)
        return focal_area_index, predicted_coords


class analyzer_logger:
    def __init__(self):
        self.stat = None
        self.csv_file_name = None
        self.is_live: bool = False  # whether the video source is the camera or mp4 file

        self.start_time = time.time() * 1000  # Convert to milliseconds

        self.log_frame = logvar(['Frames', 'time'], ['', 'ms'])
        self.log_frame.Frames = 0

        self.left_eye = logvar(
            [
                'eDistL',
                'left_pupil_x', 'left_pupil_y', 'left_pupil_z',
                'left_pupil_scr_x', 'left_pupil_scr_y',
                'left_proj_point_x', 'left_proj_point_y', 'left_proj_point_z',
                'left_proj_point_scr_x', 'left_proj_point_scr_y',
                'left_gaze_vector_x', 'left_gaze_vector_y', 'left_gaze_vector_z',
                'left_gaze_vector_r', 'left_gaze_vector_p', 'left_gaze_vector_t',
                'left_gaze_vector_scr_x', 'left_gaze_vector_scr_y',
                'left_gaze_vector_scr_r', 'left_gaze_vector_scr_t',
            ],
            [
                'cm',  # eDistL
                '', '', '',  # left_pupil (xyz)
                'pixels', 'pixels',  # left_pupil (on screen)
                '', '', '',  # left_proj_point (xyz)
                'pixels', 'pixels',  # left_proj_point (on screen)
                '', '', '',  # left_gaze_vector (xyz)
                '', 'rad', 'rad',  # left_gaze_vector (rpt)
                'pixels', 'pixels',  # left_gaze_vector (on screen) (xy)
                'pixels', 'rad',  # left_gaze_vector (on screen) (rt)
            ]
        )
        self.right_eye = logvar(
            [
                'eDistR',
                'right_pupil_x', 'right_pupil_y', 'right_pupil_z',
                'right_pupil_scr_x', 'right_pupil_scr_y',
                'right_proj_point_x', 'right_proj_point_y', 'right_proj_point_z',
                'right_proj_point_scr_x', 'right_proj_point_scr_y',
                'right_gaze_vector_x', 'right_gaze_vector_y', 'right_gaze_vector_z',
                'right_gaze_vector_r', 'right_gaze_vector_p', 'right_gaze_vector_t',
                'right_gaze_vector_scr_x', 'right_gaze_vector_scr_y',
                'right_gaze_vector_scr_r', 'right_gaze_vector_scr_t',
            ],
            [
                'cm',  # eDistR
                '', '', '',  # right_pupil (xyz)
                'pixels', 'pixels',  # right_pupil (on screen)
                '', '', '',  # right_proj_point (xyz)
                'pixels', 'pixels',  # right_proj_point (on screen)
                '', '', '',  # right_gaze_vector (xyz)
                '', '', '',  # right_gaze_vector (rpt)
                'pixels', 'pixels',  # right_gaze_vector (on screen) (xy)
                'pixels', 'rad',  # right_gaze_vector (on screen) (rt)
            ]
        )
        self.dgaze = logvar(['dGaze_x', 'dGaze_y'], ['pixels', 'pixels'])
        self.pupils = logvar(['pupils_dist'], ['pixels'])

        self.scr_dist = logvar(['scr_dist'], ['cm'])
        self.head_pose = logvar(['head_pose_x', 'head_pose_y'], ['', ''])

    def reset_time(self):
        self.start_time = time.time() * 1000  # Convert to milliseconds

    def init_csv_output(self, file_name):
        self.csv_file_name = file_name

        # check if the file exist and create an empty one if not
        self.is_live = False
        if not os.path.exists(file_name):
            with open(file_name, "w") as file:
                self.is_live = True

        # get csv file
        self.stat = csv_statistics(file_name, self.is_live)
        self.__init_stat_variables()
        self.reset_time()

    def close(self):
        if self.stat is not None:
            self.stat.close()
            self.stat = None

    def __init_stat_variables(self):
        # if video source is camera add frame counter to log
        if self.is_live:
            self.stat.add_log(self.log_frame)

        # common variable to log
        self.stat.add_log(self.left_eye)
        self.stat.add_log(self.right_eye)
        self.stat.add_log(self.dgaze)
        self.stat.add_log(self.pupils)
        self.stat.add_log(self.scr_dist)
        self.stat.add_log(self.head_pose)

    def update(self, dc: DataCollection):
        if self.is_live:
            self.log_frame.Frames += 1
            self.log_frame.time = int(time.time() * 1000 - self.start_time)

        """ Left eye """
        self.left_eye.eDistL = np.round(dc.get('depth_left_cm'), 1)
        # left_pupil
        self.left_eye.left_pupil_x = dc.get('left_pupil_x')
        self.left_eye.left_pupil_y = dc.get('left_pupil_y')
        self.left_eye.left_pupil_z = dc.get('left_pupil_z')
        self.left_eye.left_pupil_scr_x = dc.get('left_pupil_scr_x')
        self.left_eye.left_pupil_scr_y = dc.get('left_pupil_scr_y')
        # left_proj_point
        self.left_eye.left_proj_point_x = dc.get('left_proj_point_x')
        self.left_eye.left_proj_point_y = dc.get('left_proj_point_y')
        self.left_eye.left_proj_point_z = dc.get('left_proj_point_z')
        self.left_eye.left_proj_point_scr_x = dc.get('left_proj_point_scr_x')
        self.left_eye.left_proj_point_scr_y = dc.get('left_proj_point_scr_y')
        # left_gaze_vector
        self.left_eye.left_gaze_vector_x = dc.get('left_gaze_vector_x')
        self.left_eye.left_gaze_vector_y = dc.get('left_gaze_vector_y')
        self.left_eye.left_gaze_vector_z = dc.get('left_gaze_vector_z')
        self.left_eye.left_gaze_vector_r = dc.get('left_gaze_vector_r')
        self.left_eye.left_gaze_vector_p = dc.get('left_gaze_vector_p')
        self.left_eye.left_gaze_vector_t = dc.get('left_gaze_vector_t')
        self.left_eye.left_gaze_vector_scr_x = dc.get('left_gaze_vector_scr_x')
        self.left_eye.left_gaze_vector_scr_y = dc.get('left_gaze_vector_scr_y')
        self.left_eye.left_gaze_vector_scr_r = dc.get('left_gaze_vector_scr_r')
        self.left_eye.left_gaze_vector_scr_t = dc.get('left_gaze_vector_scr_t')

        """ Right eye """
        self.right_eye.eDistR = np.round(dc.get('depth_right_cm'), 1)
        # right_pupil
        self.right_eye.right_pupil_x = dc.get('right_pupil_x')
        self.right_eye.right_pupil_y = dc.get('right_pupil_y')
        self.right_eye.right_pupil_z = dc.get('right_pupil_z')
        self.right_eye.right_pupil_scr_x = dc.get('right_pupil_scr_x')
        self.right_eye.right_pupil_scr_y = dc.get('right_pupil_scr_y')
        # right_proj_point
        self.right_eye.right_proj_point_x = dc.get('right_proj_point_x')
        self.right_eye.right_proj_point_y = dc.get('right_proj_point_y')
        self.right_eye.right_proj_point_z = dc.get('right_proj_point_z')
        self.right_eye.right_proj_point_scr_x = dc.get('right_proj_point_scr_x')
        self.right_eye.right_proj_point_scr_y = dc.get('right_proj_point_scr_y')
        # right_gaze_vector
        self.right_eye.right_gaze_vector_x = dc.get('right_gaze_vector_x')
        self.right_eye.right_gaze_vector_y = dc.get('right_gaze_vector_y')
        self.right_eye.right_gaze_vector_z = dc.get('right_gaze_vector_z')
        self.right_eye.right_gaze_vector_r = dc.get('right_gaze_vector_r')
        self.right_eye.right_gaze_vector_p = dc.get('right_gaze_vector_p')
        self.right_eye.right_gaze_vector_t = dc.get('right_gaze_vector_t')
        self.right_eye.right_gaze_vector_scr_x = dc.get('right_gaze_vector_scr_x')
        self.right_eye.right_gaze_vector_scr_y = dc.get('right_gaze_vector_scr_y')
        self.right_eye.right_gaze_vector_scr_r = dc.get('right_gaze_vector_scr_r')
        self.right_eye.right_gaze_vector_scr_t = dc.get('right_gaze_vector_scr_t')

        """ Common for both eyes """
        self.dgaze.dGaze_x = dc.get('dGaze_x')
        self.dgaze.dGaze_y = dc.get('dGaze_y')

        self.pupils.pupils_dist = dc.get('pupils_dist')

        self.head_pose.head_pose_x = dc.get('head_pose_x')
        self.head_pose.head_pose_y = dc.get('head_pose_y')

        self.scr_dist.scr_dist = np.round(dc.get('scr_dist'), 1)

    def stat_push(self):
        if self.stat is not None:
            self.stat.push()
