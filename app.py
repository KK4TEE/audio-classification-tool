import os
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, BooleanVar, DoubleVar
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyaudio
import soundfile as sf
import webbrowser
import shutil
import threading

# Load configuration
config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)


# Function to load audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


# Function to plot waveform
def plot_waveform(audio, sr, frame):
    fig, ax = plt.subplots(figsize=(5, 2))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Waveform")
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# Function to plot MFCC
def plot_mfcc(audio, sr, frame):
    fig, ax = plt.subplots(figsize=(5, 2))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title("MFCC")
    fig.colorbar(librosa.display.specshow(mfccs, x_axis='time', ax=ax), ax=ax)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# Function to classify audio file
def classify_audio(file_path, label, secondary=False):
    output_path = config['secondary_output_directory'] if secondary else config['output_directory']
    output_path = os.path.join(output_path, label)
    os.makedirs(output_path, exist_ok=True)
    new_file_path = os.path.join(output_path, os.path.basename(file_path))

    status_message = ""
    if os.path.exists(new_file_path):
        result = messagebox.askquestion("File Exists", "The file already exists. Do you want to overwrite it?",
                                        icon='warning')
        if result == 'yes':
            os.remove(new_file_path)
            shutil.move(file_path, new_file_path)
            status_message = f"Success: File overwritten successfully.\n{file_path} -> {new_file_path}"
        else:
            delete_result = messagebox.askquestion("Delete Original", "Do you want to delete the original file?",
                                                   icon='warning')
            if delete_result == 'yes':
                os.remove(file_path)
                status_message = f"Deleted: The original file has been deleted.\n{file_path}"
    else:
        shutil.move(file_path, new_file_path)
        status_message = f"Success: File moved successfully.\n{file_path} -> {new_file_path}"

    app.update_status(status_message)


# Function to play audio using PyAudio with volume control
def play_audio_with_pyaudio(file_path, lock, volume):
    with lock:
        audio, sr = sf.read(file_path)
        audio = audio * (volume / 100.0)  # Adjust volume
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
        stream.write(audio.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()


def threaded_play_audio(file_path, lock, volume):
    threading.Thread(target=play_audio_with_pyaudio, args=(file_path, lock, volume)).start()


# Custom "About" dialog class
class AboutDialog(simpledialog.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="About")

    def body(self, master):
        about_text = ("Audio Classifier\n"
                      "Seth Persigehl\n")
        tk.Label(master, text=about_text, justify=tk.LEFT).pack(anchor=tk.W)

        url = "https://persigehl.com/audio-classifier"
        link = tk.Label(master, text=url, fg="blue", cursor="hand2")
        link.pack(anchor=tk.W)
        link.bind("<Button-1>", lambda e: webbrowser.open(url))

        version_text = "Version: 1.0.0\n© 2024 Seth Persigehl"
        tk.Label(master, text=version_text, justify=tk.LEFT).pack(anchor=tk.W)

    def buttonbox(self):
        box = tk.Frame(self)
        tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE).pack(pady=5)
        self.bind("<Return>", self.ok)
        box.pack()


# Custom "Keyboard Shortcuts" dialog class
class ShortcutsDialog(simpledialog.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Keyboard Shortcuts")

    def body(self, master):
        shortcuts_text = (
            "Keyboard Shortcuts:\n"
            "Ctrl+L: Load Directory\n"
            "Space: Play Audio\n"
            "Right Arrow: Next File\n"
            "Left Arrow: Previous File\n"
            "Enter: Classify as Default\n"
            "Ctrl+U: Classify as Unknown\n"
            "Ctrl+A: About\n"
            "Ctrl+K: Keyboard Shortcuts Help\n"
        )
        tk.Label(master, text=shortcuts_text, justify=tk.LEFT).pack(anchor=tk.W)

    def buttonbox(self):
        box = tk.Frame(self)
        tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE).pack(pady=5)
        self.bind("<Return>", self.ok)
        box.pack()


# GUI Application
class AudioClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Classifier")
        self.geometry(config['resolution'])

        self.auto_play_sound = BooleanVar(value=config.get('auto_play_sound', True))
        self.audio_play_lock = threading.Lock()  # Lock for controlling audio playback
        self.is_refreshing = False  # Flag to track if the list is being refreshed
        self.file_count = 0  # To keep track of the number of files in the list
        self.volume = DoubleVar(value=100)  # Volume control variable
        self.setup_ui()
        self.current_file = None
        self.audio = None
        self.sr = None
        self.initial_directory = None
        self.file_path_map = {}  # To map display paths to actual file paths

        # Force a redraw of the GUI
        self.update_idletasks()

        # Automatically load directory if valid
        self.auto_load_directory()

        # Bind keyboard shortcuts
        self.bind("<Control-l>", lambda event: self.load_directory())
        self.bind("<space>", lambda event: self.play_audio())
        self.bind("<Right>", lambda event: self.quick_action('next'))
        self.bind("<Left>", lambda event: self.quick_action('previous'))
        self.bind("<Return>", lambda event: self.classify_default())
        self.bind("<Control-u>", lambda event: self.quick_action('unknown'))
        self.bind("<Control-a>", lambda event: self.show_about())
        self.bind("<Control-k>", lambda event: self.show_shortcuts())

    def setup_ui(self):
        self.create_menu_bar()

        self.pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=10)
        self.pane.pack(fill=tk.BOTH, expand=True)

        self.file_frame = tk.Frame(self.pane, width=config['file_panel_width'])
        self.file_frame.pack_propagate(False)
        self.pane.add(self.file_frame)

        self.visualization_pane = tk.PanedWindow(self.pane, orient=tk.VERTICAL, sashwidth=10)
        self.pane.add(self.visualization_pane)

        self.visualization_frame = tk.Frame(self.visualization_pane)
        self.visualization_pane.add(self.visualization_frame, height=400)

        self.header_frame = tk.Frame(self.visualization_frame)
        self.header_frame.pack(side=tk.TOP, fill=tk.X)

        self.waveform_frame = tk.Frame(self.visualization_frame)
        self.waveform_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mfcc_frame = tk.Frame(self.visualization_frame)
        self.mfcc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.volume_frame = tk.Frame(self.visualization_frame)
        self.volume_frame.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(self.volume_frame, text="Volume").pack()
        self.volume_slider = tk.Scale(self.volume_frame, from_=200, to=0, orient=tk.VERTICAL, variable=self.volume)
        self.volume_slider.pack(fill=tk.Y, expand=True)

        self.classification_frame = tk.Frame(self.visualization_pane)
        self.visualization_pane.add(self.classification_frame)

        self.file_list_frame = tk.Frame(self.file_frame, width=config['file_panel_width'])
        self.file_list_frame.pack(fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.file_frame, width=config['file_panel_width'])
        self.button_frame.pack(fill=tk.X)

        self.load_files_button = tk.Button(self.button_frame, text="Load Directory", command=self.load_directory)
        self.load_files_button.pack(side=tk.LEFT)

        self.organize_by_file_button = tk.Button(self.button_frame, text="Organize by File Name",
                                                 command=self.organize_by_file_name)
        self.organize_by_file_button.pack(side=tk.RIGHT)

        self.organize_by_path_button = tk.Button(self.button_frame, text="Organize by Path",
                                                 command=self.organize_by_path_name)
        self.organize_by_path_button.pack(side=tk.RIGHT)

        self.scrollbar_y = tk.Scrollbar(self.file_list_frame, orient=tk.VERTICAL)
        self.scrollbar_x = tk.Scrollbar(self.file_list_frame, orient=tk.HORIZONTAL)

        self.file_list = tk.Listbox(self.file_list_frame, yscrollcommand=self.scrollbar_y.set,
                                    xscrollcommand=self.scrollbar_x.set, width=int(config['file_panel_width'] / 10))
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_list.config(justify=tk.RIGHT)

        self.scrollbar_y.config(command=self.file_list.yview)
        self.scrollbar_x.config(command=self.file_list.xview)

        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_list.bind('<<ListboxSelect>>', self.on_file_select)

        # Quick actions
        self.quick_action_frame = tk.Frame(self.classification_frame)
        self.quick_action_frame.pack(fill=tk.X, pady=10)
        for action in config['quick_actions']:
            color = "light green" if action == "play" else "light coral" if action == "unknown" else "gray80"
            if action == "accept_default":
                self.default_classify_button = tk.Button(self.quick_action_frame, text="Accept Default: None",
                                                         command=self.classify_default, height=3, bg="light green")
                self.default_classify_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            else:
                btn = tk.Button(self.quick_action_frame, text=action.capitalize(),
                                command=lambda a=action: self.quick_action(a), height=3, bg=color)
                btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Primary classifications
        self.primary_classification_frame = tk.Frame(self.classification_frame)
        self.primary_classification_frame.pack(fill=tk.X, pady=10)
        self.add_buttons_to_frame(self.primary_classification_frame, config['labels']['primary'], self.classify)

        # Secondary classifications
        self.secondary_classification_frame = tk.Frame(self.classification_frame)
        self.secondary_classification_frame.pack(fill=tk.X, pady=10)
        self.add_buttons_to_frame(self.secondary_classification_frame, config['labels']['secondary'],
                                  lambda l: self.classify(l, secondary=True))

        # Status line
        self.status_line = tk.Text(self, height=1, wrap=tk.WORD, state='disabled')
        self.status_line.pack(fill=tk.X, side=tk.BOTTOM)

    def add_buttons_to_frame(self, frame, labels, command, max_per_row=4):
        row_frame = tk.Frame(frame)
        row_frame.pack(fill=tk.X, pady=5)
        count = 0
        for label in labels:
            if count >= max_per_row:
                row_frame = tk.Frame(frame)
                row_frame.pack(fill=tk.X, pady=5)
                count = 0
            btn = tk.Button(row_frame, text=label, command=lambda l=label: command(l), height=2, width=20)
            btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            count += 1

    def create_menu_bar(self):
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Directory", command=self.load_directory)
        file_menu.add_separator()
        file_menu.add_checkbutton(label="Auto Play Sound", onvalue=True, offvalue=False, variable=self.auto_play_sound,
                                  command=self.toggle_auto_play_sound)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menu_bar)

    def show_about(self):
        AboutDialog(self)

    def show_shortcuts(self):
        ShortcutsDialog(self)

    def auto_load_directory(self):
        input_directory = config.get('input_directory')
        if (input_directory and os.path.exists(input_directory)):
            self.load_directory(input_directory)
            self.initial_directory = input_directory

    def load_directory(self, directory=None):
        if directory is None:
            directory = filedialog.askdirectory(initialdir=config['input_directory'])
        if directory:
            self.is_refreshing = True  # Set the refreshing flag
            self.initial_directory = directory
            self.file_list.delete(0, tk.END)
            self.file_paths = []
            self.file_path_map = {}  # Clear the mapping dictionary
            self.parent_path_len = len(directory) + 1
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        file_path = os.path.join(root, file)
                        self.file_paths.append(file_path)
            self.load_classified_files()
            self.organize_by_file_name()
            self.file_count = len(self.file_paths)  # Update the file count after loading the directory
            self.is_refreshing = False  # Reset the refreshing flag

    def load_classified_files(self):
        for root, _, files in os.walk(config['output_directory']):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)

    def organize_by_file_name(self):
        self.is_refreshing = True  # Set the refreshing flag
        # Save the current scroll position
        current_scroll_pos = self.file_list.yview()

        self.file_list.delete(0, tk.END)
        self.file_path_map.clear()  # Clear the mapping dictionary
        file_names = sorted(self.file_paths, key=lambda x: os.path.basename(x))
        for file in file_names:
            display_path = file[self.parent_path_len:] if self.initial_directory in file else os.path.relpath(file,
                                                                                                              config[
                                                                                                                  'output_directory'])
            self.file_list.insert(tk.END, display_path)
            self.file_path_map[display_path] = file  # Map display path to actual file path
            if config['output_directory'] in file:
                self.file_list.itemconfig(tk.END, {'fg': 'grey'})
        if self.file_list.size() > 0:
            self.file_list.select_set(0)
            self.file_list.event_generate("<<ListboxSelect>>")
        self.file_list.xview_moveto(1)

        # Reapply the saved scroll position if the number of files is similar and more than 0
        if self.file_list.size() > 0 and abs(self.file_list.size() - self.file_count) <= 2:
            self.file_list.yview_moveto(current_scroll_pos[0])

        self.is_refreshing = False  # Reset the refreshing flag

    def organize_by_path_name(self):
        self.is_refreshing = True  # Set the refreshing flag
        # Save the current scroll position
        current_scroll_pos = self.file_list.yview()

        self.file_list.delete(0, tk.END)
        self.file_path_map.clear()  # Clear the mapping dictionary
        path_names = sorted(self.file_paths)
        for file in path_names:
            display_path = file[self.parent_path_len:] if self.initial_directory in file else os.path.relpath(file,
                                                                                                              config[
                                                                                                                  'output_directory'])
            self.file_list.insert(tk.END, display_path)
            self.file_path_map[display_path] = file  # Map display path to actual file path
            if config['output_directory'] in file:
                self.file_list.itemconfig(tk.END, {'fg': 'grey'})
        if self.file_list.size() > 0:
            self.file_list.select_set(0)
            self.file_list.event_generate("<<ListboxSelect>>")
        self.file_list.xview_moveto(1)

        # Reapply the saved scroll position if the number of files is similar and more than 0
        if self.file_list.size() > 0 and abs(self.file_list.size() - self.file_count) <= 2:
            self.file_list.yview_moveto(current_scroll_pos[0])

        self.is_refreshing = False  # Reset the refreshing flag

    def on_file_select(self, event):
        selected_index = self.file_list.curselection()
        if selected_index:
            display_path = self.file_list.get(selected_index)
            selected_file = self.file_path_map[display_path]
            self.current_file = selected_file
            self.audio, self.sr = load_audio(selected_file)
            self.update_visualizations()
            self.update_default_button()

    def play_audio(self):
        if self.current_file:
            threaded_play_audio(self.current_file, self.audio_play_lock, self.volume.get())

    def update_visualizations(self):
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()
        for widget in self.mfcc_frame.winfo_children():
            widget.destroy()

        # Add header above waveforms
        for widget in self.header_frame.winfo_children():
            widget.destroy()

        file_name = os.path.basename(self.current_file)
        file_extension = os.path.splitext(file_name)[1]
        file_path = os.path.dirname(self.current_file)
        header_text = f"File: {file_path}/{file_name} ({file_extension})"

        header_label = tk.Label(self.header_frame, text=header_text, font=("Helvetica", 12, "bold"))
        header_label.pack(side=tk.TOP, fill=tk.X)

        plot_waveform(self.audio, self.sr, self.waveform_frame)
        plot_mfcc(self.audio, self.sr, self.mfcc_frame)

    def update_default_button(self):
        if self.current_file:
            default_label = os.path.basename(os.path.dirname(self.current_file))
            self.default_classify_button.config(text=f"Accept Default: {default_label}")

    def classify(self, label, secondary=False):
        if self.current_file and label:
            current_index = self.file_list.curselection()[0]
            current_scroll_pos = self.file_list.yview()
            classify_audio(self.current_file, label, secondary)
            next_index = min(current_index + 1, self.file_list.size() - 1)
            self.load_directory(self.initial_directory)
            self.file_list.select_clear(0, tk.END)
            self.file_list.select_set(next_index)
            self.file_list.event_generate("<<ListboxSelect>>")
            self.file_list.yview_moveto(current_scroll_pos[0])

    def classify_default(self):
        if self.current_file:
            default_label = os.path.basename(os.path.dirname(self.current_file))
            self.classify(default_label)

    def quick_action(self, action):
        if action == 'play':
            self.play_audio()
        elif action == 'next':
            current_index = self.file_list.curselection()
            if current_index:
                next_index = current_index[0] + 1
                if next_index < self.file_list.size():
                    self.file_list.select_clear(0, tk.END)
                    self.file_list.select_set(next_index)
                    self.file_list.event_generate("<<ListboxSelect>>")
        elif action == 'previous':
            current_index = self.file_list.curselection()
            if current_index:
                prev_index = current_index[0] - 1
                if prev_index >= 0:
                    self.file_list.select_clear(0, tk.END)
                    self.file_list.select_set(prev_index)
                    self.file_list.event_generate("<<ListboxSelect>>")
        elif action == 'accept_default':
            self.classify_default()
        elif action == 'unknown':
            if self.current_file:
                current_index = self.file_list.curselection()[0]
                current_scroll_pos = self.file_list.yview()
                unknown_path = os.path.join(config['output_directory'], 'unknown')
                os.makedirs(unknown_path, exist_ok=True)
                shutil.move(self.current_file, os.path.join(unknown_path, os.path.basename(self.current_file)))
                next_index = min(current_index + 1, self.file_list.size() - 1)
                self.load_directory(self.initial_directory)
                self.file_list.select_clear(0, tk.END)
                self.file_list.select_set(next_index)
                self.file_list.event_generate("<<ListboxSelect>>")
                self.file_list.yview_moveto(current_scroll_pos[0])

    def toggle_auto_play_sound(self):
        config['auto_play_sound'] = self.auto_play_sound.get()
        with open(config_file_path, 'w') as file:
            yaml.safe_dump(config, file)

    def update_status(self, message):
        self.status_line.config(state='normal')
        self.status_line.delete(1.0, tk.END)
        self.status_line.insert(tk.END, message)
        self.status_line.config(state='disabled')
        # Clear the status message after 5 seconds
        self.after(5000, self.clear_status)

    def clear_status(self):
        self.status_line.config(state='normal')
        self.status_line.delete(1.0, tk.END)
        self.status_line.config(state='disabled')


if __name__ == "__main__":
    app = AudioClassifierApp()
    app.mainloop()
