import time
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from helper.helper import *


def calc_stat_for_valid_data(csv_data, param_dict_list):
    """
    csv_data - CSV data dict
    param_dict_list - list of param dictionary.
                      each element contain:
                      - name: parameter base name (without unit)
                      - mean: mean list for each valid interval
                      - std: std list for each valid interval
                      - x: the x position in screen plane for the n'th valid interval
                      - y: the y position in screen plane for the n'th valid interval
    """
    processed_dict = {key.split('[')[0]: value for key, value in csv_data.items()}
    valid = np.array(processed_dict['valid'])

    for param_dict in param_dict_list:
        param_dict['mean'] = []
        param_dict['std'] = []
        param_dict['x'] = []
        param_dict['y'] = []

    # Calculate mean and variance
    pulse_start = None

    for i in range(len(valid)):
        if valid[i] == 1:
            if pulse_start is None:
                pulse_start = i
        elif pulse_start is not None:
            # End of a valid pulse
            for param_dict in param_dict_list:
                data = processed_dict[param_dict['name']]

                pulse_data = np.array(data[pulse_start:i])
                pulse_data = pulse_data[np.isfinite(pulse_data)]

                param_dict['mean'].append(pulse_data.mean())
                param_dict['std'].append(pulse_data.std())
                param_dict['x'].append(processed_dict['x'][pulse_start])
                param_dict['y'].append(processed_dict['y'][pulse_start])

            pulse_start = None

    # Check if there's an ongoing pulse at the end of the array
    if pulse_start is not None:
        for param_dict in param_dict_list:
            data = processed_dict[param_dict['name']]

            pulse_data = np.array(data[pulse_start:i])
            pulse_data = pulse_data[np.isfinite(pulse_data)]

            param_dict['mean'].append(pulse_data.mean())
            param_dict['std'].append(pulse_data.std())
            param_dict['x'].append(processed_dict['x'][pulse_start])
            param_dict['y'].append(processed_dict['y'][pulse_start])


def create_grid_data(data_dict, screen_width, screen_height, rows, cols):
    """
    Creates grid-based statistics from scattered calibration data.

    Args:
        data_dict: Dictionary containing:
            - mean: List of mean values for each calibration point
            - std: List of std values for each calibration point
            - x: List of x positions
            - y: List of y positions
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        rows: Number of grid rows
        cols: Number of grid columns

    Returns:
        Dictionary with grid-based statistics and bounding boxes.
    """
    # Convert input data to numpy arrays
    x = np.array(data_dict['x'])
    y = np.array(data_dict['y'])
    means = np.array(data_dict['mean'])
    stds = np.array(data_dict['std'])

    # Create grid coordinates for interpolation (cols x rows)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), num=cols),
        np.linspace(y.min(), y.max(), num=rows)
    )

    # Interpolate mean and std values across the screen plane
    grid_means = griddata(
        (x, y), means,
        (grid_x, grid_y), method='cubic', fill_value=0
    )
    grid_stds = griddata(
        (x, y), stds,
        (grid_x, grid_y), method='cubic', fill_value=0
    )

    # Calculate grid cell dimensions
    cell_width = screen_width / cols
    cell_height = screen_height / rows

    # Initialize output data structure
    grid_data = {
        'mean': [],
        'std': [],
        'area_bbox': []
    }

    # Assign interpolated values to each cell directly
    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries
            x_start = int(col * cell_width)
            y_start = int(row * cell_height)
            bbox = (x_start, y_start, int(cell_width), int(cell_height))

            # Get the interpolated mean and std for this cell
            cell_mean = grid_means[row, col]
            cell_std = grid_stds[row, col]

            # Append data to the output dictionary
            grid_data['mean'].append(cell_mean)
            grid_data['std'].append(cell_std)
            grid_data['area_bbox'].append(bbox)

    return grid_data


def visualize_grid_data(screen_width, screen_height, n_rows, n_cols, focused_cell=None, ci_status=None):
    # Create a black background
    if ci_status is None or ci_status == 'NO' or ci_status == 'NA':
        frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 50
    elif ci_status == 'CI':
        frame = np.full((screen_height, screen_width, 3), (80, 150, 80), dtype=np.uint8)
    else:  # ci_status == 'DI':
        frame = np.full((screen_height, screen_width, 3), (180, 120, 60), dtype=np.uint8)

    # Calculate cell dimensions
    cell_width = screen_width // n_cols
    cell_height = screen_height // n_rows

    # Generate grid boundaries
    grid_data = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * cell_width
            y = row * cell_height
            grid_data.append((x, y, cell_width, cell_height))

    # Draw cells
    for idx, (bbox_x, bbox_y, bbox_w, bbox_h) in enumerate(grid_data):
        if focused_cell is not None and focused_cell == idx:
            cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), CV2_COLOR_RED, -1)
        else:
            cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), CV2_COLOR_WHITE, 2)

    return frame


class AverageSimple:
    def __init__(self, window_size: int):
        self.window_size: int = window_size
        self.window = np.array([])
        self._current_val: float = 0.0

    def add(self, new_value):
        self.window = np.append(self.window, new_value)

        if self.window.size >= self.window_size:
            self._current_val = self.window.sum() / self.window.size
            self.window = np.array([])

    def get(self):
        return self._current_val


class MovingAverageSimple:
    def __init__(self, window_size: int):
        self.window_size: int = window_size
        self.window = np.array([])
        self._current_val: float = 0.0

    def add(self, new_value):
        self.window = np.concatenate((self.window, new_value), axis=0)[-self.window_size:]

        if self.window.size >= self.window_size:
            self._current_val = self.window.sum() / self.window.size

    def get(self):
        return self._current_val


class MovingWindow:
    def __init__(self, window_size: int, func):
        self.window_size: int = window_size
        self.window = np.array([])
        self.func = func
        self._current_val: float = 0.0

    def add(self, new_value):
        self.window = np.concatenate((self.window, new_value), axis=0)[-self.window_size:]

        if self.window.size >= self.window_size:
            self._current_val = self.func(self.window)

    def get(self):
        return self._current_val

    def get_last(self):
        return self.window[-1]


class ClassificationAverage:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.window = np.array([], dtype=int)
        self._current_class = 0

    def add(self, new_class: int) -> int:
        self.window = np.concatenate([self.window, [new_class]], dtype=int)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        values, counts = np.unique(self.window, return_counts=True)
        self._current_class = values[np.argmax(counts)]

        return self._current_class

    def get(self) -> int:
        return self._current_class


class CI_Classification:
    def __init__(self, window_size: int = 5,
                 value_for_true: float = 20,
                 threshold_for_detect: float = 0.4):
        self.window_size = window_size
        self.value_for_true = value_for_true
        self.threshold_for_detect = threshold_for_detect
        self.window = np.array([])
        self._current_class = False

    def add(self, new_value: float) -> bool:
        self.window = np.concatenate([self.window, [new_value]])
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        # Count how many values in the window are above value_for_true
        true_count = np.sum(self.window > self.value_for_true)

        # Calculate the proportion of true values
        true_proportion = true_count / len(self.window)
        self._current_class = true_proportion >= self.threshold_for_detect

        return self._current_class

    def get(self) -> bool:
        return self._current_class

    def get_all(self):
        return self.window > self.value_for_true


class CI_RealTimeDetect:
    def __init__(self,
                 window_size: int,
                 thresholds: np.ndarray,
                 threshold_for_alert: float,
                 save_path: str = '',
                 save_window_size: int = 1000):
        """
        Initialize the detector.

        @param window_size: Size of the history window.
        @param thresholds: numpy array of thresholds for each focal area.
        @param threshold_for_alert: Alert if (detection_count / window_size) >= this value.
        @param save_path: File path to save the large window data as CSV.
        @param save_window_size: Size of the large window for saving data.
        """
        self.window_size = window_size
        self.thresholds = thresholds
        self.threshold_for_alert = threshold_for_alert
        self.window = np.array([], dtype=bool)  # Will store True/False for detections
        self._alert = False

        # For saving stats
        self.save_path = save_path
        self.save_window_size = save_window_size
        self.save_data = []
        self.start_time = None
        self.frame_count = 0

    def add(self, focal_area: int, ci_dist: float) -> bool:
        """
        Add a new detection.

        @param focal_area: Index of the focal area (0 to n-1).
        @param ci_dist: Value to compare with threshold.
        @return: True if alert is triggered, False otherwise.
        """
        # Check if ci_dist is above the threshold for this focal_area
        detected = ci_dist > self.thresholds[focal_area]
        if detected:
            print(f"detected({self.frame_count}): {ci_dist} > {self.thresholds[focal_area]}")
        self.window = np.concatenate([self.window, [detected]])

        # Keep window size fixed
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        if len(self.save_path) != 0:
            # Save data for large window
            self.save_data.append({
                'frame[]': self.frame_count,
                'time[sec]': time.time(),
                'SPCD[mm]': ci_dist,
                'threshold[mm]': self.thresholds[focal_area],
                'focal_area[]': focal_area,
            })
            self.frame_count += 1

            # If save_data reached save_window_size, save to CSV and clear
            if len(self.save_data) >= self.save_window_size:
                self.save_stat()

        # Calculate if alert should be triggered
        detection_count = np.sum(self.window)
        if len(self.window) == self.window_size:
            alert_proportion = detection_count / len(self.window)
            self._alert = alert_proportion >= self.threshold_for_alert
            if self._alert:
                print(f"alert: {alert_proportion} >= {self.threshold_for_alert}")
        return self._alert

    def save_stat(self):
        """
        Save the collected data to CSV file and clear the saved data list.
        """
        df = pd.DataFrame(self.save_data)
        # Append to CSV file, create if not exists
        df.to_csv(self.save_path, mode='a', header=not pd.io.common.file_exists(self.save_path), index=False)
        self.save_data = []

    def get(self) -> bool:
        return self._alert


class RealtimeLogger:
    def __init__(self, save_path: str, save_window_size: int = 100):
        """
        @param save_path: File path to save the large window data as CSV.
        @param save_window_size: Size of the large window for saving data.
        """
        self.save_path = save_path
        self.save_window_size = save_window_size
        self.save_data = []
        self.start_time = None
        self.frame_count = 0

    def add(self, data_dict):
        # Save data for large window
        self.save_data.append({
            'frame[]': self.frame_count,
            'time[sec]': time.time(),
            **data_dict
        })
        self.frame_count += 1

        # If save_data reached save_window_size, save to CSV and clear
        if len(self.save_data) >= self.save_window_size:
            self.save_stat()

    def save_stat(self):
        df = pd.DataFrame(self.save_data)
        # Append to CSV file, create if not exists
        df.to_csv(self.save_path, mode='a', header=not pd.io.common.file_exists(self.save_path), index=False)
        self.save_data = []
