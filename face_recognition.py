import cv2
import os
import numpy as np
import pickle
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import winsound  # for beep alert (Windows)

# ---------------- Paths ----------------
faces_dir = "faces_db"
model_file = "face_recognizer.yml"
names_file = "face_names.pkl"

if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

# ---------------- Haar Cascade ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- LBPH Recognizer ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(model_file) and os.path.exists(names_file):
    recognizer.read(model_file)
    with open(names_file, "rb") as f:
        names = pickle.load(f)
else:
    names = {}

# ---------------- Tkinter GUI ----------------
root = tk.Tk()
root.title("Face Recognition System")

# Left: video feed
video_label = tk.Label(root)
video_label.grid(row=0, column=0, rowspan=10)

# ---------------- Functions ----------------
def add_person():
    global adding_face, new_name, samples_collected
    new_name = simpledialog.askstring("Add Person", "Enter name of new person:")
    if new_name:
        adding_face = True
        samples_collected = 0

def delete_person():
    global recognizer, names
    person = simpledialog.askstring("Delete Person", "Enter name to delete:")
    if person:
        # Remove images
        for filename in os.listdir(faces_dir):
            if filename.startswith(person + "_"):
                os.remove(os.path.join(faces_dir, filename))
        # Retrain recognizer if faces remain
        remaining_files = os.listdir(faces_dir)
        if remaining_files:
            training_images, training_labels = [], []
            temp_names = {}
            for idx, filename in enumerate(remaining_files):
                img = cv2.imread(os.path.join(faces_dir, filename), cv2.IMREAD_GRAYSCALE)
                training_images.append(img)
                training_labels.append(idx)
                temp_names[idx] = filename.split("_")[0]
            recognizer.train(training_images, np.array(training_labels))
            recognizer.save(model_file)
            with open(names_file, "wb") as f:
                pickle.dump(temp_names, f)
            names = temp_names
            messagebox.showinfo("Delete Person", f"{person} deleted successfully.")
        else:
            # No faces left
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(names_file): os.remove(names_file)
            names = {}
            messagebox.showinfo("Delete Person", f"{person} deleted. No faces left.")

def exit_app():
    root.destroy()

# Buttons on right
tk.Button(root, text="Add Person", command=add_person, width=20).grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Delete Person", command=delete_person, width=20).grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Exit", command=exit_app, width=20).grid(row=2, column=1, padx=10, pady=5)

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)
adding_face = False
new_name = ""
samples_collected = 0
max_samples = 20

# ---------------- Update Frame ----------------
def update_frame():
    global adding_face, samples_collected, new_name, recognizer, names
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        name = "Unknown"

        if len(names) > 0 and not adding_face:
            try:
                label_id, confidence = recognizer.predict(roi_gray)
                if confidence < 80:
                    name = names.get(label_id, "Unknown")
            except:
                name = "Unknown"

        # BEEP alert for new faces
        if name == "Unknown" and not adding_face:
            winsound.Beep(1000, 200)

        # Adding new face
        if adding_face and new_name:
            samples_collected += 1
            file_path = os.path.join(faces_dir, f"{new_name}_{samples_collected}.png")
            cv2.imwrite(file_path, roi_gray)
            cv2.putText(frame, f"Saving {new_name} ({samples_collected}/{max_samples})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if samples_collected >= max_samples:
                # Retrain recognizer
                training_images, training_labels = [], []
                names = {}
                for idx, filename in enumerate(os.listdir(faces_dir)):
                    img = cv2.imread(os.path.join(faces_dir, filename), cv2.IMREAD_GRAYSCALE)
                    training_images.append(img)
                    training_labels.append(idx)
                    person_name = filename.split("_")[0]
                    names[idx] = person_name
                recognizer.train(training_images, np.array(training_labels))
                recognizer.save(model_file)
                with open(names_file, "wb") as f:
                    pickle.dump(names, f)
                adding_face = False
                samples_collected = 0
                new_name = ""

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Convert OpenCV frame to ImageTk
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ---------------- Start ----------------
update_frame()
root.mainloop()
cap.release()
