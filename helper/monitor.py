from threading import Lock

import numpy as np
import screeninfo
import time


def get_laptop_monitor_number():
    monitors = screeninfo.get_monitors()
    return 0


class monitor:
    def __init__(self, inst=0):
        monitors = screeninfo.get_monitors()
        self.info = monitors[inst]
        self.w_pixel_size = self.info.width / self.info.width_mm  # [pixel/mm]
        self.h_pixel_size = self.info.height / self.info.height_mm  # [pixel/mm]

    def width_mm2pixels(self, mm):
        return self.w_pixel_size * mm

    def height_mm2pixels(self, mm):
        return self.h_pixel_size * mm

    def width_pixels2mm(self, pixels):
        return pixels / self.w_pixel_size

    def height_pixels2mm(self, pixels):
        return pixels / self.h_pixel_size

    def print_all(self):
        print(f"Width: {self.info.width_mm} mm")
        print(f"Height: {self.info.height_mm} mm")
        print(f"Resolution: {self.info.width}x{self.info.height} pixels")
        print(f"DPI: {self.info.width / (self.info.width_mm / 25.4):.2f}")


class scr_point:
    def __init__(self, x=0, y=0, unit: str = "", additional: str = None):
        self.x = x
        self.y = y
        self.unit = unit
        self.additional = additional
        self.mtx_lock = Lock()

    def set(self, x, y):
        with self.mtx_lock:
            self.x = x
            self.y = y

    def set_int(self, x, y):
        with self.mtx_lock:
            self.x = int(x)
            self.y = int(y)

    def show(self):
        with self.mtx_lock:
            print(f"({self.x},{self.y})")

    def get_log_header(self):
        if self.additional is not None:
            return [f'x{self.additional}[{self.unit}]', f'y{self.additional}[{self.unit}]']
        else:
            return [f'x[{self.unit}]', f'y[{self.unit}]']

    def get_log(self):
        with self.mtx_lock:
            return [self.x, self.y]

    def __eq__(self, other):
        with self.mtx_lock:
            if not isinstance(other, scr_point):
                return False

            return (self.x, self.y, self.unit, self.additional) == (other.x, other.y, other.unit, other.additional)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        with self.mtx_lock:
            return f"({self.x}, {self.y}, {self.unit}, {self.additional})"


def get_focal_area_pos(predicted_x, predicted_y, screen_width, screen_height, grid_rows, grid_cols):

    """
    Determines the index of the grid cell containing the predicted gaze point.

    The screen is divided into a grid (grid_rows x grid_cols).
    Cell indexing starts at 0 (top-left) and increases row by row,
    then column by column.

    Args:
        predicted_x (float): The predicted X coordinate of the gaze point (in pixels).
        predicted_y (float): The predicted Y coordinate of the gaze point (in pixels).
        screen_width (int): The total width of the screen in pixels.
        screen_height (int): The total height of the screen in pixels.
        grid_rows (int): The number of rows in the grid overlaying the screen.
        grid_cols (int): The number of columns in the grid overlaying the screen.

    Returns:
        tuple: col and row index.
    """
    # --- Calculate Cell Dimensions ---
    # Add a small epsilon to prevent potential floating point issues at the exact boundaries
    epsilon = 1e-9
    cell_width = screen_width / grid_cols
    cell_height = screen_height / grid_rows

    if cell_width <= 0 or cell_height <= 0:
        print("Error: Calculated cell dimensions are non-positive.")
        return -1  # Should not happen with positive inputs, but good check

    # --- Determine Row and Column Index ---
    # Use integer division to find the index. Clamp coordinates to be within screen bounds first.
    clamped_x = max(0, min(predicted_x, screen_width - epsilon))
    clamped_y = max(0, min(predicted_y, screen_height - epsilon))

    col_index = int(clamped_x // cell_width)
    row_index = int(clamped_y // cell_height)

    # Ensure indices are within valid grid range (e.g., if clamped_x was exactly screen_width)
    # This clamping step after calculation might be redundant due to clamping coords first,
    # but provides an extra layer of safety.
    col_index = min(col_index, grid_cols - 1)
    row_index = min(row_index, grid_rows - 1)

    return col_index, row_index


def get_focal_area_index(predicted_x, predicted_y, screen_width, screen_height, grid_rows, grid_cols):
    """
    Determines the index of the grid cell containing the predicted gaze point.

    The screen is divided into a grid (grid_rows x grid_cols).
    Cell indexing starts at 0 (top-left) and increases row by row,
    then column by column.

    Args:
        predicted_x (float): The predicted X coordinate of the gaze point (in pixels).
        predicted_y (float): The predicted Y coordinate of the gaze point (in pixels).
        screen_width (int): The total width of the screen in pixels.
        screen_height (int): The total height of the screen in pixels.
        grid_rows (int): The number of rows in the grid overlaying the screen.
        grid_cols (int): The number of columns in the grid overlaying the screen.

    Returns:
        int: The index of the focused cell (0 to grid_rows * grid_cols - 1),
             or -1 if inputs are invalid.
    """
    # --- Calculate Cell Dimensions ---
    col_index, row_index = get_focal_area_pos(
        predicted_x, predicted_y,
        screen_width, screen_height,
        grid_rows, grid_cols
    )

    # --- Calculate Final Cell Index ---
    # Index = (row number * number of columns per row) + column number
    focus_cell_index = (row_index * grid_cols) + col_index

    return focus_cell_index


class performance:
    def __init__(self, *, window_size: int = 10):
        self.window_size = window_size
        self.dt = np.array([])
        self.begin = time.time()

    def get_dt(self):
        end = time.time()
        dt = end - self.begin
        self.begin = end

        self.dt = np.concatenate([self.dt, [dt]])
        self.dt = self.dt[-self.window_size:]
        return self.dt.mean()

    def get_fps(self):
        return 1 / self.get_dt()
