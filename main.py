import os
import sys
import tkinter as tk
from ui.app_window import AppWindow


def main():
    video_file = "video.dav"
    if not os.path.exists(video_file):
        print(f"Atenção: Arquivo '{video_file}' não encontrado em {os.getcwd()}")

    root = tk.Tk()
    app = AppWindow(root, video_path=video_file)
    root.mainloop()


if __name__ == "__main__":
    main()
