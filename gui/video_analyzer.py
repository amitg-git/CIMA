from tkinter import filedialog
from cima.cima import CIMA


class video_analyzer_screen:
    def __init__(self, prev_screen, cima: CIMA, view_analyze):
        self.prev_screen = prev_screen
        self.cima = cima
        self.view_analyze = view_analyze

        self.setup_ui()

    def on_closing(self):
        self.prev_screen.deiconify()  # Show the prev screen again

    def setup_ui(self):
        # select mp4 file
        filetypes = (('MP4 files', '*.mp4'), )
        filenames = filedialog.askopenfilenames(title='Select an MP4 File', initialdir='captures', filetypes=filetypes)
        if filenames:
            for i, filename in enumerate(filenames):
                print(f"Analyze files: {i+1}/{len(filenames)}")
                self.cima.reset()
                manual_exit_flag = self.cima.analyze(filename, view_analyze=self.view_analyze)
                if manual_exit_flag:
                    break

        self.on_closing()
