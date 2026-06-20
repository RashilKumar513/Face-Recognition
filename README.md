# Face Recognition

A simple desktop face recognition application implemented in Python using OpenCV and a small Tkinter GUI. This project captures faces from your webcam, trains an LBPH face recognizer, and recognizes faces in real-time. It is intended as an educational/demo project.

## Key features
- Real-time face detection using OpenCV's Haar Cascade
- Face recognition using LBPH (Local Binary Patterns Histograms)
- Add new persons by capturing multiple face samples via webcam
- Delete person entries and retrain the recognizer
- Simple Tkinter GUI for controls and live video preview
- Saves trained model and label mapping for persistence

## Files and data
- `face_recognition.py` — main application script (GUI + recognition logic)
- `faces_db/` — directory where captured face images are stored (created automatically)
- `face_recognizer.yml` — trained LBPH model file (created after training)
- `face_names.pkl` — mapping of label IDs to person names (created after training)
- `README.md` — this file

## How it works (overview of the code)
1. On startup the app ensures `faces_db/` exists and attempts to load a previously saved LBPH model (`face_recognizer.yml`) and names mapping (`face_names.pkl`). If available, the recognizer is ready to predict.
2. The webcam feed is captured using OpenCV (cv2.VideoCapture(0)). Frames are converted to grayscale and faces are detected using Haar Cascade (`haarcascade_frontalface_default.xml`).
3. If the Add Person button is used, the app prompts for a name then collects up to 20 samples of the detected face, saving each as `NAME_index.png` under `faces_db/`.
4. Once the sample quota is reached the app loads all images from `faces_db/`, assigns consecutive numeric labels to each image, trains an LBPH recognizer on these images, saves the trained model (`face_recognizer.yml`) and the label-name mapping (`face_names.pkl`).
5. During normal operation, for each detected face the LBPH recognizer predicts a label and a confidence score; if confidence is below a threshold (80 in this code) the system displays the corresponding name, otherwise it shows "Unknown" and plays a beep sound (Windows-only via winsound).
6. The Delete Person button removes any saved images that start with the given name and retrains the model from remaining images (or deletes model files if none remain).

## Requirements
- Python 3.8+ (tested on Windows)
- A webcam

Python packages (install using pip):
- opencv-python
- opencv-contrib-python
- numpy
- pillow

On Windows the code uses `winsound` (built-in) to play a beep when an unknown face is detected. On other OSes you can remove or replace the beep call (see Troubleshooting).

Example pip install command:

pip install opencv-python opencv-contrib-python numpy pillow

If you prefer a requirements file, create `requirements.txt` with the packages above and run `pip install -r requirements.txt`.

## Run the application
1. Clone or download the repository and open a terminal in the project directory.
2. (Recommended) Create and activate a virtual environment:

   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

3. Install dependencies (see Requirements section).
4. Run the app:

   python face_recognition.py

The Tkinter window opens and shows the live camera feed. Use the buttons on the right:
- Add Person — prompts for a name, then collects 20 face samples from the camera and auto-trains the recognizer.
- Delete Person — prompts for a name and deletes any images starting with that name, then retrains (or removes model files if none left).
- Exit — closes the app and releases the camera.

## Tips for capturing good samples
- Make sure the face is well-lit and facing the camera.
- Capture from multiple angles and expressions for better recognition robustness.
- Avoid large variations in background; crop is done using face bounding boxes.

## How to customize
- Number of samples per person: change `max_samples` variable (default 20).
- Confidence threshold for recognition: change the value `if confidence < 80:` (lower value = stricter match).
- Haar cascade: the script uses OpenCV's default frontal face cascade. For better results try alternative cascades or DNN-based detectors.
- Beep behavior: replace `winsound.Beep(1000, 200)` with a cross-platform solution (for example, playsound or simple sound libraries) or remove it.

## Troubleshooting
- No camera found: ensure camera is connected and accessible by other apps; try changing `cv2.VideoCapture(0)` index to `1` or another index if multiple cameras are present.
- ImportError for cv2: install `opencv-python` and `opencv-contrib-python`.
- Permission errors on camera: on macOS and Windows ensure terminal/app has camera permission.
- Model fails to load: if `face_recognizer.yml` or `face_names.pkl` are missing or corrupted delete them and re-add persons to retrain.

## Security & privacy
- The project stores raw face images in `faces_db/`. Treat this directory as sensitive data and do not commit it to public repositories.
- Consider hashing filenames, encrypting storage, or storing only face embeddings in production systems.

## Limitations
- This is a simple demo — LBPH is lightweight but less accurate than modern deep-learning-based face models.
- The alert uses Windows-only `winsound`. Cross-platform sound support is not included out-of-the-box.
- No authentication, access control, or secure storage is implemented.

## License
Include a license of your choice (for example MIT). Add a `LICENSE` file to the repository if you want to allow others to re-use the code.

## Contact
Repository: https://github.com/RashilKumar513/Face-Recognition

If you want, I can also:
- Add a requirements.txt and a LICENSE file
- Replace the Windows-only beep with a cross-platform solution
- Add CLI flags (capture count, camera index, threshold) or a simple settings panel

