import tkinter as tk
from config import Config
from gui.gui import LoginScreen


def main():
    root = tk.Tk()
    root.iconbitmap(Config.APP_ICON)
    app = LoginScreen(root)
    root.mainloop()


if __name__ == "__main__":
    main()
