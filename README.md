# Real-Time Face Recognition

A simple, real-time face recognition project built with Python, OpenCV, and the `face_recognition` library. This application uses your computer's webcam to detect and identify known faces based on a directory of reference images.

## Features

- **Real-Time Detection:** Captures video from your webcam and processes it in real-time.
- **Easy to Add Faces:** Simply drop an image of a person into the `images/` folder, name the file as the person's name (e.g., `Elon Musk.jpg`), and the system will automatically recognize them.
- **Optimized for Speed:** Video frames are resized down during processing to ensure a smooth framerate.

## Prerequisites

- Python 3.x
- A working webcam attached to your computer

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repository-url>
   cd Face_Recognition
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Create a virtual environment named "face"
   python -m venv face
   
   # Activate it on Linux/macOS
   source face/bin/activate
   
   # Or on Windows
   face\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The requirements specifically lock down `setuptools<70.0.0` to prevent an import error with the underlying `face_recognition_models` package).*

## How to Use

1. **Add Reference Images:**
   Place images of the people you want to recognize inside the `images/` directory. The name of the file (without the extension) will be used as the displayed name.
   - Example: `images/Bill Gates.jpg` will output **"Bill Gates"** when detected.

2. **Run the Script:**
   Make sure your virtual environment is active, then start the main script:
   ```bash
   python main.py
   ```

3. **Stop the Application:**
   To quit out of the live camera feed, press the **`Esc`** key on your keyboard.

## File Structure

- `main.py`: The main entry point. It manages the webcam operations and draws bounding boxes around detected faces.
- `face_rec.py`: Contains the `Facerec` class, handling the background logic—loading reference face encodings and comparing them against incoming frames.
- `images/`: The directory where you store your known dataset images. 
- `requirements.txt`: Project dependencies list.
- `.gitignore`: Configured to protect secrets and ignore compiled/cache files.
