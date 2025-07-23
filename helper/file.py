import os
import csv
import time
from datetime import datetime
from typing import List, Any
from threading import Lock
import numpy as np


def increment_filename(filename):
    name, ext = os.path.splitext(filename)
    base, *rest = name.rsplit('_', 1)
    num = int(rest[0]) + 1 if rest and rest[0].isdigit() else 1
    return f"{base}_{num}{ext}"


def get_unique_filename(filename):
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = increment_filename(new_filename)
    return new_filename


def get_dated_filename(filename, format_spec="%Y%m%d_%H%M%S", sep="_"):
    """
    Append timestamp to filename while preserving extension

    Parameters:
    - filename: Original filename (str) - e.g., 'log.txt', '/data/report.csv'
    - format_spec: datetime format string (default: YYYYMMDD_HHMMSS)
    - sep: Separator between name and timestamp (default: '_')

    Returns: Dated filename (str)

    Examples:
    >>> get_dated_filename('log.txt')
    'log_20231025_143022.txt'

    >>> get_dated_filename('data.json', format_spec='%Y-%m-%d')
    'data_2023-10-25.json'
    """
    # Split into base and extension
    base, ext = os.path.splitext(filename)

    # Generate timestamp string
    timestamp = datetime.now().strftime(format_spec)

    # Construct new filename
    return f"{base}{sep}{timestamp}_1{ext}"


class csv_logger:
    def __init__(self, filename: str, user_id):
        self.writer = None
        self.file = None
        self.user_id = user_id
        self.filename = get_unique_filename(f"captures/{user_id}_{filename}_1.csv")

        self.log_sources = []
        self.Frames = 0
        self.start_time = time.time() * 1000  # Convert to milliseconds
        self.header_written = False
        self.header = ['Frames', 'Time[ms]']

    def open(self):
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

    def add_log(self, log_source: Any):
        if hasattr(log_source, 'get_log') and callable(getattr(log_source, 'get_log')) and \
                hasattr(log_source, 'get_log_header') and callable(getattr(log_source, 'get_log_header')):
            self.log_sources.append(log_source)
            self.header.extend(log_source.get_log_header())
        else:
            raise AttributeError(f"The provided log source does not have required methods")

    def push(self):
        log_data = []
        for log_source in self.log_sources:
            log_data.extend(log_source.get_log())

        if log_data:  # Only proceed if there's data to log
            self.Frames += 1
            current_time = time.time() * 1000 - self.start_time  # Time since init in ms

            if not self.header_written:
                self.writer.writerow(self.header)
                self.header_written = True

            row = [self.Frames, f"{current_time:.3f}"]
            row.extend(log_data)
            self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.close()


class csv_statistics:
    def __init__(self, file_name: str, is_realtime: bool = False):
        self.file_name = file_name
        self.is_realtime = is_realtime
        self.log_sources = []
        self.header = []
        self.data_indx = 0

        # Load existing data from the CSV file
        self.data = []
        if os.path.exists(file_name):
            with open(file_name, mode='r', newline='') as file:
                reader = csv.reader(file)
                if not self.is_realtime:
                    self.header = next(reader)  # Read the header
                    self.data = list(reader)  # Read the rest of the data
                else:
                    self.header = []
                    self.data = []
                file.close()

    def add_log(self, log_source: Any):
        if (hasattr(log_source, 'get_log') and callable(getattr(log_source, 'get_log')) and
                hasattr(log_source, 'get_log_header') and callable(getattr(log_source, 'get_log_header'))):
            self.log_sources.append(log_source)
            self.header.extend(log_source.get_log_header())
        else:
            raise AttributeError("The provided log source does not have required methods")

    def push(self):
        # Prepare new row based on log sources
        if not self.is_realtime:
            for log_source in self.log_sources:
                new_values = log_source.get_log()
                self.data[self.data_indx].extend(new_values)

            # go to next row
            self.data_indx += 1
        else:
            # create the new row
            self.data.append([])

            for log_source in self.log_sources:
                new_values = log_source.get_log()
                self.data[-1].extend(new_values)

    def close(self):
        # Save the updated data to a new CSV file in the Statistics folder
        output_file = os.path.join("Statistics", os.path.basename(self.file_name))

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.header)  # Write header
            writer.writerows(self.data)  # Write data
            file.close()


class logvar:
    def __init__(self, variables=None, units=None):
        if variables is None:
            variables = []
        if units is None:
            units = []

        if len(variables) != len(units):
            raise ValueError("The number of variables must match the number of units")

        self.mtx_lock = Lock()

        self.data = {var: 0 for var in variables}
        self.units = dict(zip(variables, units))

    def set(self, **kwargs):
        with self.mtx_lock:
            for var, value in kwargs.items():
                if var in self.data:
                    self.data[var] = value
                else:
                    raise ValueError(f"Variable '{var}' is not defined for this {self.__class__.__name__}")

    def get_log_header(self):
        with self.mtx_lock:
            return [f'{var}[{unit}]' for var, unit in self.units.items()]

    def get_log(self):
        with self.mtx_lock:
            return list(self.data.values())

    def list(self):
        with self.mtx_lock:
            return self.get_log()

    def array(self):
        with self.mtx_lock:
            return np.array(self.get_log())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        with self.mtx_lock:
            return ", ".join(f"{var}={value}[{self.units[var]}]" for var, value in self.data.items())

    def __getattr__(self, name):
        if name in self.data:
            with self.mtx_lock:
                return self.data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if 'data' in self.__dict__ and name in self.data:
            with self.mtx_lock:
                self.data[name] = value
        elif name in ['data', 'units']:
            super().__setattr__(name, value)
        elif name == 'mtx_lock':  # Special case for mtx_lock
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class FileLogger:
    def __init__(self, file_path, *,
                 is_unique: bool = True,
                 cont_log: bool = False,
                 also_print: bool = True,
                 only_print: bool = False):
        """
        Initialize logger with file path
        :param file_path: Path to log file (e.g., 'app.log')
        """
        self.also_print = also_print
        self.cont_log = cont_log
        self.only_print = only_print
        self.file_path = None

        if not self.only_print:
            self.file_path = get_dated_filename(file_path) if is_unique else file_path
            self.file_path = get_unique_filename(self.file_path)
            self._init_file()

    def _init_file(self):
        """Initialize/clear log file"""
        mod = 'a' if self.cont_log else 'w'
        with open(self.file_path, mod) as f:
            f.write('')  # Create empty file or clear existing

    def log(self, *args, **kwargs):
        """
        Write to log file like print()
        Supports all print() arguments:
        - Multiple values (log("Hello", "World"))
        - sep= (separator)
        - end= (line ending)
        """
        # Convert all arguments to strings
        output = []

        # Handle positional arguments
        if args:
            output.append(kwargs.get('sep', ' ').join(str(arg) for arg in args))

        # Handle keyword arguments (except sep/end/file)
        remaining_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ('sep', 'end', 'file')}
        if remaining_kwargs:
            output.append(str(remaining_kwargs))

        # Write with proper line ending
        line_ending = kwargs.get('end', '\n')

        text = ''.join(output) + line_ending

        if self.only_print or self.also_print:
            print(text, end='')

        if not self.only_print:
            with open(self.file_path, 'a') as f:
                f.write(text)
