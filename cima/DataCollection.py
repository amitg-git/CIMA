import re
import time

import joblib
import numpy as np
import pandas as pd
from pyvista import cartesian_to_spherical as xyz_to_rpt


def xy_to_rt(x, y):
    r = np.hypot(x, y)  # Equivalent to sqrt(x² + y²)
    theta = np.arctan2(y, x)  # Angle in radians (-π to π)
    return r, theta


class DataCollection:
    def __init__(self, gaze_model_features_path, ci_model_features_path):
        self.init_scr_dist = np.inf

        self.datadict: dict = {
            #########################
            # """ Distance Algo """ #
            #########################
            'depth_left_cm': np.inf,
            'depth_right_cm': np.inf,
            'scr_dist': np.inf,

            ##########################
            # """ LaserGaze Algo """ #
            ##########################
            # left_pupil
            'left_pupil_x': np.inf, 'left_pupil_y': np.inf, 'left_pupil_z': np.inf,
            'left_pupil_r': np.inf, 'left_pupil_p': np.inf, 'left_pupil_t': np.inf,
            'left_pupil_scr_x': np.inf, 'left_pupil_scr_y': np.inf,
            # left_proj_point
            'left_proj_point_x': np.inf, 'left_proj_point_y': np.inf, 'left_proj_point_z': np.inf,
            'left_proj_point_r': np.inf, 'left_proj_point_p': np.inf, 'left_proj_point_t': np.inf,
            'left_proj_point_scr_x': np.inf, 'left_proj_point_scr_y': np.inf,
            # left_gaze_vector
            'left_gaze_vector_scr_x': np.inf, 'left_gaze_vector_scr_y': np.inf,
            'left_gaze_vector_scr_r': np.inf, 'left_gaze_vector_scr_t': np.inf,
            'left_gaze_vector_x': np.inf, 'left_gaze_vector_y': np.inf, 'left_gaze_vector_z': np.inf,
            'left_gaze_vector_r': np.inf, 'left_gaze_vector_p': np.inf, 'left_gaze_vector_t': np.inf,
            # right_pupil
            'right_pupil_x': np.inf, 'right_pupil_y': np.inf, 'right_pupil_z': np.inf,
            'right_pupil_r': np.inf, 'right_pupil_p': np.inf, 'right_pupil_t': np.inf,
            'right_pupil_scr_x': np.inf, 'right_pupil_scr_y': np.inf,
            # right_proj_point
            'right_proj_point_x': np.inf, 'right_proj_point_y': np.inf, 'right_proj_point_z': np.inf,
            'right_proj_point_r': np.inf, 'right_proj_point_p': np.inf, 'right_proj_point_t': np.inf,
            'right_proj_point_scr_x': np.inf, 'right_proj_point_scr_y': np.inf,
            # right_gaze_vector
            'right_gaze_vector_scr_x': np.inf, 'right_gaze_vector_scr_y': np.inf,
            'right_gaze_vector_scr_r': np.inf, 'right_gaze_vector_scr_t': np.inf,
            'right_gaze_vector_x': np.inf, 'right_gaze_vector_y': np.inf, 'right_gaze_vector_z': np.inf,
            'right_gaze_vector_r': np.inf, 'right_gaze_vector_p': np.inf, 'right_gaze_vector_t': np.inf,
            # delta gaze
            'dGaze_x': np.inf, 'dGaze_y': np.inf,
            # Pupillary distance
            'pupils_dist': np.inf,
            #####################
            # """ gaze algo """ #
            #####################
            'head_pose_x': np.inf, 'head_pose_y': np.inf,

            ###############################
            # """ gaze focus ML model """ #
            ###############################
            # 'focal_x': 0, 'focal_y': 0
            'focal_area_x': 0, 'focal_area_y': 0
        }
        # self.gaze_model_base_columns = ['focal_x', 'focal_y']
        self.gaze_model_base_columns = ['focal_area_x', 'focal_area_y']
        self.base_columns = list(self.datadict.keys())
        for key in self.gaze_model_base_columns:
            self.base_columns.remove(key)

        self.gaze_model_features = joblib.load(gaze_model_features_path)
        self.ci_model_features = joblib.load(ci_model_features_path)

        gaze_model_history_depth = self._get_history_depth(self.gaze_model_features)
        ci_model_features = self._get_history_depth(self.ci_model_features)
        self.history_depth = max(gaze_model_history_depth, ci_model_features)
        self._add_history()

    def get(self, col):
        return self.datadict[col]

    def _get_history_depth(self, keys):
        key = keys[0]
        key_pattern = re.compile(rf'^{re.escape(key)}_n(\d+)$')
        max_n = 0
        for col in keys:
            match = key_pattern.match(col)
            if match:
                n = int(match.group(1))
                if n > max_n:
                    max_n = n
        return max_n + 1

    def _add_history(self):
        if self.history_depth <= 1:
            return

        lagged_cols = []
        for col in self.datadict.keys():
            for n in range(1, self.history_depth):
                lagged_col = f"{col}_n{n}"
                lagged_cols.append(lagged_col)

        for col in lagged_cols:
            self.datadict[col] = np.inf

    def __calc_concluded_data(self):
        if self.init_scr_dist == np.inf:
            return

        def df_to_vec(df, vec_name, list_axis):
            vec = np.array([df[f'{vec_name}_{axis}'] for axis in list_axis])
            return vec

        def vec_to_df(df, vec_name, input_vec, axis='xy'):
            list_axis = list(axis)
            for i, axis in enumerate(list_axis):
                df[f'{vec_name}_{axis}'] = input_vec[i]

        def sub_df_vecs(df, vec_names, vec_res=None, axis='xy'):
            list_axis = list(axis)
            vec0 = df_to_vec(df, vec_names[0], list_axis)
            vec1 = df_to_vec(df, vec_names[1], list_axis)
            res = vec0 - vec1

            if vec_res is None:
                return res

            vec_to_df(df, vec_res, res, axis)

        def to_xyz(arr):
            return arr[0], arr[1], arr[2]

        def to_xy(arr):
            return arr[0], arr[1]

        def df_cartesian_to_polar(df, vec_name, dim):
            base_axis = 'xy' if dim == 2 else 'xyz'
            target_axis = 'rt' if dim == 2 else 'rtp'

            vec_cartesian = df_to_vec(df, vec_name, base_axis)
            if dim == 2:
                vec_polar = xy_to_rt(*to_xy(vec_cartesian))
            else:
                vec_polar = xyz_to_rpt(*to_xyz(vec_cartesian))

            vec_to_df(df, vec_name, vec_polar, target_axis)

        sub_df_vecs(self.datadict, ['left_proj_point_scr', 'left_pupil_scr'], 'left_gaze_vector_scr', axis='xy')
        sub_df_vecs(self.datadict, ['right_proj_point_scr', 'right_pupil_scr'], 'right_gaze_vector_scr', axis='xy')

        df_cartesian_to_polar(self.datadict, 'left_gaze_vector_scr', dim=2)
        df_cartesian_to_polar(self.datadict, 'right_gaze_vector_scr', dim=2)

        df_cartesian_to_polar(self.datadict, 'left_gaze_vector', dim=3)
        df_cartesian_to_polar(self.datadict, 'right_gaze_vector', dim=3)

        self.datadict['pupils_dist'] = np.linalg.norm(sub_df_vecs(self.datadict, ['right_pupil_scr', 'left_pupil_scr']))

    def _update_history(self, cols):
        # For each base column, prepare the lagged columns
        for key in cols:
            for i in range(self.history_depth - 1, 1, -1):
                self.datadict[f"{key}_n{i}"] = self.datadict[f"{key}_n{i - 1}"]
            self.datadict[f"{key}_n1"] = self.datadict[key]

    def update(self, new_dict: dict):
        # Lock init screen distance from user id data is valid
        if self.init_scr_dist == np.inf:
            self.init_scr_dist = new_dict['scr_dist']

        # Update history pipeline
        if self.history_depth > 1:
            self._update_history(self.base_columns)

        # Update current data Collection
        for col in new_dict.keys():
            self.datadict[col] = new_dict[col]

        # Update concluded data
        self.__calc_concluded_data()

    def update_focal_xy(self, col: int, row: int):
        # Update history pipeline
        if self.history_depth > 1:
            self._update_history(self.gaze_model_base_columns)

        # Update current data Collection
        self.datadict['focal_x'] = col
        self.datadict['focal_y'] = row

    def update_focal_area(self, col: int, row: int):
        # Update history pipeline
        if self.history_depth > 1:
            self._update_history(self.gaze_model_base_columns)

        # Update current data Collection
        self.datadict['focal_area_x'] = col
        self.datadict['focal_area_y'] = row

    def get_gaze_model_features(self):
        filtered_dict = {k: self.datadict[k] for k in self.gaze_model_features if k in self.datadict}
        return filtered_dict

    def get_ci_model_features(self):
        filtered_dict = {k: self.datadict[k] for k in self.ci_model_features if k in self.datadict}
        return filtered_dict
