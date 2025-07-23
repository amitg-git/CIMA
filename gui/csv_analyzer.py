import os
import re
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
from tkinterdnd2 import DND_FILES, TkinterDnD
from config import Config
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from itertools import zip_longest, cycle

from cima.cima import CIMA
from helper.statistics import calc_stat_for_valid_data


class csv_analyzer_screen:
    def __init__(self, prev_screen: tk.Tk, cima: CIMA):
        self.moving_avg_entry = None
        self.dc_free_flag = False
        self.only_valid_flag = False
        self.dc_free_btn = None
        self.only_valid_btn = None
        self.focus_points_view_btn = None
        self.update_plot_button = None
        self.scale_value_entry = None
        self.scale_graph_dropdown = None
        self.y_axis_listbox = None
        self.data = None
        self.axis_frame = None
        self.y_axis_dropdown = None
        self.x_axis_dropdown = None
        self.ax = None
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.instruction_label = None
        self.menu_frame = None
        self.content_frame = None
        self.file_buttons = []
        self.current_file = None
        self.initial_xlim = None
        self.initial_ylim = None

        self.cima = cima

        self.prev_screen = prev_screen
        self.prev_screen.withdraw()  # Hide the main screen

        self.master = TkinterDnD.Tk()
        self.master.title(Config.APP_TITLE)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)  # Set up the window close event

        self.center_window("csv_analyzer")
        self.setup_ui()
        self.setup_navigation()

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
        self.menu_frame = tk.Frame(self.master, width=Config.CSV_ANALYZER_MENU_WIDTH, bg='lightgray')
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.menu_frame.pack_propagate(False)  # Prevent the frame from shrinking

        # Create a frame for the content
        self.content_frame = tk.Frame(self.master)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Add a label for instructions
        self.instruction_label = tk.Label(self.content_frame, text="Drag and drop a CSV file here", bg="lightblue")
        self.instruction_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Enable drag and drop for the instruction label
        self.instruction_label.drop_target_register(DND_FILES)
        self.instruction_label.dnd_bind('<<Drop>>', self.on_drop)

        # Create a matplotlib figure and canvas
        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.content_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add dropdown menus for X and Y axis selection
        self.axis_frame = tk.Frame(self.content_frame)
        self.axis_frame.pack(pady=10)

        # list of graphs to plot
        tk.Label(self.axis_frame, text="X-axis:").grid(row=0, column=1, padx=5)
        self.x_axis_dropdown = ttk.Combobox(self.axis_frame)
        self.x_axis_dropdown.grid(row=0, column=2, padx=5)

        tk.Label(self.axis_frame, text="Y-axis:").grid(row=0, column=3, padx=5)
        self.y_axis_listbox = tk.Listbox(self.axis_frame, selectmode=tk.MULTIPLE, exportselection=0, width=30)
        self.y_axis_listbox.grid(row=0, column=4, padx=5)

        # manipulate 1 graph only
        tk.Label(self.axis_frame, text="scale-graph").grid(row=1, column=1, padx=5)
        self.scale_graph_dropdown = ttk.Combobox(self.axis_frame)
        self.scale_graph_dropdown.grid(row=1, column=2, padx=5)

        # Register the validation function
        validate_float_cmd = self.axis_frame.register(self.validate_float)
        validate_int_cmd = self.axis_frame.register(self.validate_int)

        tk.Label(self.axis_frame, text="scale-value").grid(row=2, column=1, padx=5)
        self.scale_value_entry = ttk.Entry(self.axis_frame, validate="key", validatecommand=(validate_float_cmd, '%P'))
        self.scale_value_entry.grid(row=2, column=2, padx=5)

        tk.Label(self.axis_frame, text="moving-average").grid(row=2, column=3, padx=5)
        self.moving_avg_entry = ttk.Entry(self.axis_frame, validate="key", validatecommand=(validate_int_cmd, '%P'))
        self.moving_avg_entry.grid(row=2, column=4, padx=5)

        # Add a button to update the plot
        self.update_plot_button = ttk.Button(self.axis_frame, text="Update Plot", command=self.update_plot)
        self.update_plot_button.grid(row=0, column=5, padx=5)

        # Add a button for DC free
        self.dc_free_btn = ttk.Button(self.axis_frame, text="DC Free", command=self.dc_free_plot)
        self.dc_free_btn.grid(row=1, column=4, padx=5)

        # Add a button for Only Valid
        self.only_valid_btn = ttk.Button(self.axis_frame, text="Only Valid", command=self.only_valid_plot)
        self.only_valid_btn.grid(row=1, column=5, padx=5)

        # Add a button for focus point view only
        self.focus_points_view_btn = ttk.Button(self.axis_frame, text="Focus Points", command=self.focus_points_view)
        self.focus_points_view_btn.grid(row=0, column=6, padx=5)

        # Bind the dropdown selection to update the plot
        self.x_axis_dropdown.bind("<<ComboboxSelected>>", self.update_plot)

    # Validation function for float input
    @staticmethod
    def validate_float(new_value):
        if new_value == "":
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_int(new_value):
        if new_value == "":
            return True
        try:
            int(new_value)
            return True
        except ValueError:
            return False

    def setup_navigation(self):
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.content_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_drop(self, event):
        file_path = event.data
        if file_path.lower().endswith('.csv'):
            self.add_file_to_menu(file_path)
            self.process_csv(file_path)
        else:
            messagebox.showerror("Error", "Please drop a CSV file.")

    def add_file_to_menu(self, file_path):
        file_name = os.path.basename(file_path)
        button = tk.Button(self.menu_frame, text=file_name,
                           command=lambda: self.process_csv(file_path))
        button.pack(fill=tk.X, padx=5, pady=2)
        button.bind('<Button-2>', lambda e: self.remove_file_from_menu(button, file_path))
        self.file_buttons.append((button, file_path))

    def remove_file_from_menu(self, button, file_path):
        button.destroy()
        self.file_buttons = [(b, f) for b, f in self.file_buttons if f != file_path]
        if self.current_file == file_path:
            self.clear_plot()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.canvas.draw()
        self.instruction_label.config(text="Drag and drop a CSV file here")
        self.current_file = None
        self.initial_xlim = None
        self.initial_ylim = None

        self.x_axis_dropdown['values'] = []
        self.x_axis_dropdown.set('')
        self.y_axis_listbox.delete(0, tk.END)

    def process_csv(self, file_path):
        try:
            with open(file_path, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)

                # Read the header row
                column_names = next(csv_reader)
                if len(column_names) < 2:
                    raise ValueError("CSV file must have at least 2 columns")

                # Update dropdown and listbox
                self.x_axis_dropdown['values'] = column_names
                self.scale_graph_dropdown['values'] = column_names
                self.y_axis_listbox.delete(0, tk.END)
                for column in column_names:
                    self.y_axis_listbox.insert(tk.END, column)

                # Set default selection for X-axis dropdown
                self.x_axis_dropdown.set(column_names[0])
                self.scale_graph_dropdown.set(column_names[0])

                # Read all rows into a list
                rows = list(csv_reader)

                # Transpose rows to columns
                columns = list(zip_longest(*rows, fillvalue=''))
                data = {col: [] for col in column_names}
                for i, name in enumerate(data):
                    data[name] = np.array(columns[i], dtype=float)

            self.data = data
            self.current_file = file_path
        except Exception as e:
            messagebox.showerror("Error", f"Error processing file: {str(e)}")

    @staticmethod
    def moving_average(data, window_size):
        # Create weights for the moving average
        weights = np.ones(window_size) / window_size

        # Perform convolution with 'valid' mode
        valid_result = np.convolve(data, weights, mode='valid')

        # Determine how many padding values are needed
        pad_size = (len(data) - len(valid_result))  # Total padding needed

        # Pad the start of the result with its first value and the end with its last value
        padded_result = np.pad(
            valid_result,  (pad_size, 0),  mode='edge'
        )
        return padded_result

    def dc_free_plot(self, event=None):
        self.dc_free_flag = not self.dc_free_flag
        self.update_plot()

    def only_valid_plot(self, event=None):
        self.only_valid_flag = not self.only_valid_flag
        self.update_plot()

    def focus_points_view(self):
        y_columns = [self.y_axis_listbox.get(i) for i in self.y_axis_listbox.curselection()]
        if len(y_columns) > 0:
            param_dict_list = [{'name': y_col.split('[')[0]} for y_col in y_columns]
            calc_stat_for_valid_data(self.data, param_dict_list)

            self.instruction_label.config(
                text=f"Plotted {', '.join(y_columns)} from {os.path.basename(self.current_file)}")

            self.plot_focus_points_view(param_dict_list)

    def plot_focus_points_view(self, param_dict_list):
        # Clear previous plot
        self.ax.clear()
        for text in self.ax.texts:
            text.remove()

        colors = cycle(plt.cm.tab10.colors)

        # Create scatter plot with labels
        for indx, param_dict in enumerate(param_dict_list):
            # Extract values from dictionary
            means = np.array(param_dict['mean'])
            stds = np.array(param_dict['std'])
            x_vals = np.array(param_dict['x'])
            y_vals = (np.array(param_dict['y']) + indx * 80)

            group_name = param_dict['name']
            cur_color = next(colors)

            #  Plot all points for the group at once to create a single legend entry
            self.ax.scatter(x_vals, y_vals, color=cur_color, label=group_name)

            # Loop again, but ONLY to add the text annotations for each point
            for mean, std, x, y in zip(means, stds, x_vals, y_vals):
                # Format label text
                mean_label_text = f"{mean:.2f}" if mean > 1 else f"{mean:.2e}"
                std_label_text = f"{std:.2f}" if std > 1 else f"{std:.2e}"
                label_text = f"{mean_label_text} ± {std_label_text}"

                # Add text label (the scatter plot itself is already done)
                self.ax.text(x, y, label_text,
                             fontsize=9,
                             verticalalignment='bottom',
                             horizontalalignment='right')

        # Add the legend after plotting all groups
        self.ax.legend()

        # Add labels and title
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        file_name = os.path.basename(self.current_file)
        self.ax.set_title(f'{file_name} - Statistical Parameter Visualization')
        self.ax.grid(True)

        # Get current limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        # Extend limits by 300 in each direction
        new_x_limits = (x_min - 300, x_max + 300)
        new_y_limits = (y_min - 300, y_max + 300)

        # Set new limits FIRST
        self.ax.set_xlim(new_x_limits)
        self.ax.set_ylim(new_y_limits)

        # Invert the axis
        self.ax.invert_yaxis()

        # Redraw canvas
        self.canvas.draw()

    def update_plot(self, event=None):
        if self.data is None:
            return

        x_column = self.x_axis_dropdown.get()
        y_columns = [self.y_axis_listbox.get(i) for i in self.y_axis_listbox.curselection()]

        sel_column = self.scale_graph_dropdown.get()
        sel_scale = float(self.scale_value_entry.get()) if self.scale_value_entry.get() else 1.0

        moving_avg = int(self.moving_avg_entry.get()) if self.moving_avg_entry.get() else 1
        moving_avg = max(moving_avg, 1)

        if x_column and y_columns:
            # Clear previous plot
            self.clear_axes()

            unique_units, y0_col, y1_col = [], [], []
            all_lines = []  # Collect all line artists

            # Plot each selected Y-column
            ax = self.ax
            colors = cycle(plt.cm.tab10.colors)
            for y_column in y_columns:
                y_data = np.array(self.moving_average(self.data[y_column], moving_avg))

                if self.only_valid_flag and 'valid[]' in self.data:
                    invalid_mask = ~(self.data['valid[]'] == 1)
                    y_data[invalid_mask] = -float('inf')

                if self.dc_free_flag:
                    inf_mask = ~np.isinf(y_data)
                    y_data -= np.mean(y_data[inf_mask])

                if y_column != sel_column or sel_scale == 1.0:
                    y_label = y_column
                else:
                    y_data *= sel_scale
                    y_label = f"{y_column}*"

                unit = re.search(r'\[(.*?)\]', y_column).group(1)
                if unit not in unique_units:
                    if len(unique_units) == 2:
                        continue
                    unique_units.append(unit)
                    if len(unique_units) == 2:
                        ax = self.ax.twinx()

                line = ax.plot(self.data[x_column], y_data, label=y_label, color=next(colors))
                all_lines.extend(line)  # Add line artist to the list

                if unit == unique_units[0]:
                    y0_col.append(y_column)
                else:
                    y1_col.append(y_column)

            # Set label to X-axes
            self.ax.set_xlabel(x_column)

            # Set labels to Y-axes
            self.ax.set_ylabel(', '.join(y0_col))
            if ax is not self.ax:
                ax.set_ylabel(', '.join(y1_col))

            # Set title
            self.ax.set_title(os.path.basename(self.current_file))
            self.ax.grid(True)

            # Create single legend for all lines
            if all_lines:
                self.ax.legend(handles=all_lines, loc='best')

            # Store initial view limits
            self.initial_xlim = self.ax.get_xlim()
            self.initial_ylim = self.ax.get_ylim()

            # Redraw canvas
            self.figure.tight_layout()
            self.canvas.draw()

            self.instruction_label.config(
                text=f"Plotted {', '.join(y_columns)} vs {x_column} from {os.path.basename(self.current_file)}")

    def clear_axes(self):
        # Get figure and all axes
        for i, other_ax in enumerate(self.figure.axes):
            if other_ax.bbox.bounds == self.ax.bbox.bounds:  # Twins share bounds
                other_ax.clear()
                if i > 0:
                    other_ax.remove()
