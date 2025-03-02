import cv2
import numpy as np
from keras.api.models import load_model
from collections import deque
from siren import playtone
import sounddevice as sd


def playtone():
    # Frequency of the beep (in Hertz)
    frequency = 510  # 1000 Hz

    # Duration of the beep (in seconds)
    duration = 1  # 0.5 seconds

    # Generate the beep sound
    sample_rate = 44100  # Samples per second (standard for audio playback)
    t = np.linspace(0, duration, int(sample_rate * duration), False)  # Time vector
    sound_wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate sine wave for beep

    # Play the sound
    for i in range(2):
        sd.play(sound_wave, sample_rate)
        sd.wait()  # Wait until the sound has finished playing


model = load_model('F:\Weapon Detection System\ANPR\WeaponPrediction.h5')




def preprocess_frame(frame, target_size):
    """
    Preprocess a single frame from the camera for model input.

    Args:
        frame (np.ndarray): Input frame from the camera.
        target_size (tuple): Target size (width, height) for resizing.

    Returns:
        np.ndarray: Preprocessed frame.
    """
    frame_resized = cv2.resize(frame, target_size)  # Resize frame
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    return frame_expanded


def predict_and_display(model, target_size, class_labels, smoothing_window=10, scale=1.5):
    """
    Use the model to predict on frames from the live camera feed and display stabilized results.

    Args:
        model (tf.keras.Model): Trained model.
        target_size (tuple): Input size for the model.
        class_labels (list): List of class labels.
        smoothing_window (int): Number of frames for smoothing bounding box predictions.
        scale (float): Factor to scale the bounding box size.
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    # Queue for storing recent bounding box predictions
    box_history = deque(maxlen=smoothing_window)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for model input
        processed_frame = preprocess_frame(frame, target_size)

        # Predict
        box_preds, class_preds = model.predict(processed_frame, verbose=0)
        class_idx = np.argmax(class_preds[0])  # Predicted class index
        class_confidence = np.max(class_preds[0])  # Confidence for class prediction

        # Scale bounding box predictions back to frame size
        h, w, _ = frame.shape
        normalized_box = box_preds[0]  # [x_min, y_min, x_max, y_max] normalized
        x_min, y_min, x_max, y_max = normalized_box * [w, h, w, h]
        x_min, x_max = int(x_min), int(x_max)
        y_min, y_max = int(y_min), int(y_max)

        # Add current bounding box to history
        box_history.append([x_min, y_min, x_max, y_max])

        # Compute the smoothed bounding box
        smoothed_box = np.mean(box_history, axis=0).astype(int)
        x_min, y_min, x_max, y_max = smoothed_box

        # Expand the bounding box by the scale factor
        box_width = x_max - x_min
        box_height = y_max - y_min
        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        new_width = box_width * scale
        new_height = box_height * scale

        x_min = max(0, int(x_center - new_width / 2))
        y_min = max(0, int(y_center - new_height / 2))
        x_max = min(w, int(x_center + new_width / 2))
        y_max = min(h, int(y_center + new_height / 2))

        # Draw stabilized and scaled bounding box
        if class_idx == 1:
            #playtone()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display class label and confidence
        if class_idx == 1:
            label = f"Weapon Detected {class_confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Show the frame
        cv2.imshow("Live Predictions (Scaled and Stabilized)", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Parameters
target_size = (224,224)
class_labels = ["Class 0", "Class 1"]

# Start live prediction
predict_and_display(model, target_size, class_labels)
