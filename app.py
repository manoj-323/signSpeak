import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk


LABEL_MAP = {
    'good evening': 0, 'good night': 1, 'thank you': 2, 'pleased': 3,
    'hello': 4, 'how are you': 5, 'alright': 6, 'good morning': 7,
    'good afternoon': 8, 'dog': 9, 'cat': 10, 'fish': 11, 'bird': 12,
    'cow': 13, 'extra': 14, 'hat': 15, 'dress': 16, 'suit': 17,
    'skirt': 18, 'shirt': 19, 'loud': 20, 'quiet': 21, 'happy': 22,
    'sad': 23, 'beautiful': 24, 'ugly': 25, 'deaf': 26, 'blind': 27
}

# Create a reverse lookup
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# Define actions
ACTIONS = np.array(['good evening',
                    'good night','thank you','pleased','hello','how are you','alright','good morning','good afternoon','dog', 'cat','fish','bird','cow','extra','hat','dress','suit','skirt','shirt','loud','quiet','happy','sad','beautiful','ugly','deaf','blind'])


# Load the trained model
model = load_model(r'4-_hp-tuned.keras')

# Initialize MediaPipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to process the frame with MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

# Function to extract keypoints
def extract_keypoints(results):
    def flatten_landmarks(landmarks, size):
        if landmarks:
            return np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
        return np.zeros(size)
    
    pose = flatten_landmarks(results.pose_landmarks, 132)  # 33*4
    face = flatten_landmarks(results.face_landmarks, 1404) # 468*3
    lh = flatten_landmarks(results.left_hand_landmarks, 63) # 21*3
    rh = flatten_landmarks(results.right_hand_landmarks, 63) # 21*3
    
    keypoints = np.concatenate([pose, face, lh, rh])
    return np.pad(keypoints, (0, 1662 - keypoints.shape[0]), 'constant')

# Tkinter App Class
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")

        # Camera label
        self.camera_label = tk.Label(root)
        self.camera_label.pack()

        # Textbox for recognized gestures
        self.text_box = tk.Text(root, height=6, width=50, font=("Arial", 16))
        self.text_box.pack(pady=10)
        self.text_box.insert(tk.END, "Recognized Gesture:\nConfidence Scores:")

        # Initialize camera and threading
        self.cap = cv2.VideoCapture(0)
        self.sequence = []
        self.last_action = None
        self.running = True

        # Start camera thread
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

        # Close event handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def process_video(self):
        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6) as holistic:
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-34:]

                if len(self.sequence) == 34:
                    res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    
                    # Get predicted action
                    predicted_index = np.argmax(res)
                    predicted_action = REVERSE_LABEL_MAP[predicted_index]

                    # Confidence scores for all actions
                    confidence_scores = {REVERSE_LABEL_MAP[i]: round(res[i] * 100, 2) for i in range(len(res))}

                    # Update text box only if action changes
                    if predicted_action != self.last_action:
                        self.last_action = predicted_action
                        self.update_text_box(predicted_action, confidence_scores)


                # Convert image to display in Tkinter
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                img = ImageTk.PhotoImage(img)

                # Update Tkinter UI
                self.camera_label.config(image=img)
                self.camera_label.image = img

    def update_text_box(self, action, confidence_scores):
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, f"Recognized Gesture: {action}\n\nConfidence Scores:\n")
        for gesture, score in confidence_scores.items():
            self.text_box.insert(tk.END, f"{gesture}: {score}%\n")

    def on_closing(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
