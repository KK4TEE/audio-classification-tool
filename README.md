# Audio classification Tool

Audio Classification Tool is a graphical user interface (GUI) application for classifying and organizing audio files into specific categories. This tool is designed to facilitate the tagging and management of audio datasets, particularly for use in machine learning projects.

## Features

- **Audio Playback**: Play selected audio files using PyAudio.
- **Waveform and MFCC Visualization**: Display the waveform and Mel-frequency cepstral coefficients (MFCC) of the selected audio file.
- **File Classification**: Classify audio files into primary and secondary categories. Move files to designated folders based on their classification.
- **Quick Actions**: Easily navigate through the audio files and classify them using quick action buttons.
- **Keyboard Shortcuts**: Perform common actions quickly using keyboard shortcuts.
- **Status Updates**: View the status of operations in the status bar at the bottom of the application.
- **Configurable Settings**: Customize the application settings through a configuration file (`config.yaml`).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/KK4TEE/audio-classification-tool.git
    cd audio-tag
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure that you have `PyAudio` installed. If not, you can install it using:
    ```sh
    pip install pyaudio
    ```

## Configuration

The application is configured using a `config.yaml` file. The configuration file should include the following settings:

- `input_directory`: Directory containing the audio files to be classified.
- `output_directory`: Directory where classified audio files will be moved.
- `secondary_output_directory`: Directory for secondary classifications.
- `labels`: Dictionary containing `primary` and `secondary` labels for classification.
- `quick_actions`: List of quick actions available in the application.
- `resolution`: Resolution of the application window.
- `file_panel_width`: Width of the file panel in the application.

## Usage

1. Run the application:
    ```sh
    python main.py
    ```

2. Load the directory containing your audio files.

3. Use the interface to play, visualize, and classify audio files:
    - Select an audio file from the list to view its waveform and MFCC.
    - Use the buttons to classify the audio file into the desired category.
    - Utilize quick action buttons for common tasks like playing the audio or moving to the next file.
    - Press `Ctrl+L` to load a new directory, `Space` to play audio, `Right Arrow` to go to the next file, and `Left Arrow` to go to the previous file.

## Keyboard Shortcuts

- `Ctrl+L`: Load Directory
- `Space`: Play Audio
- `Right Arrow`: Next File
- `Left Arrow`: Previous File
- `Enter`: Classify as Default
- `Ctrl+U`: Classify as Unknown
- `Ctrl+A`: About
- `Ctrl+K`: Keyboard Shortcuts Help

---

© 2024 Seth Persigehl
