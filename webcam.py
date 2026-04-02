import cv2
import numpy as np
import tensorflow as tf
from tkinter import filedialog, Tk
import os

# --- CONFIGURATION ---
MODEL_PATH = 'eye_disease_model.keras'
# IMPORTANT: Ensure this matches the exact alphabetical order of your data folders
CLASS_NAMES = ['Bulging_Eyes', 'Cataracts', 'Conjunctivitis', 'Crossed_Eyes', 
               'Glaucoma', 'Normal', 'Pterygium', 'Uveitis']

# 1. Load the trained model
try:
    print("Loading model... please wait.")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: Could not load model. Ensure '{MODEL_PATH}' exists.")
    exit()

def preprocess_image(image, target_size=(224, 224)):
    """Prepares any image (webcam frame or uploaded file) for the model"""
    resized = cv2.resize(image, target_size)
    img_array = tf.keras.utils.img_to_array(resized)
    img_array = tf.expand_dims(img_array, 0) 
    return img_array

def get_prediction(img_array):
    """Runs model and returns label and confidence"""
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    label = CLASS_NAMES[class_idx]
    confidence = np.max(predictions[0]) * 100
    return label, confidence

# --- CONTINUOUS MAIN LOOP ---
while True:
    print("\n" + "="*40)
    print("      EYE DISEASE DETECTION SYSTEM      ")
    print("="*40)
    print("1. Live Webcam Mode")
    print("2. Upload Image File")
    print("3. Exit Program")
    
    choice = input("\nEnter choice (1, 2, or 3): ")

    if choice == '1':
        # --- WEBCAM MODE ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            continue
            
        print("\nWebcam Active. Press 'q' on the video window to stop.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1) # Mirror effect
            
            # Predict
            proc_img = preprocess_image(frame)
            label, conf = get_prediction(proc_img)

            # UI Overlay
            color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)
            
            # Header background for text
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(frame, f"RESULT: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"CONFIDENCE: {conf:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Small window display
            cv2.imshow('Live Eye Detection', cv2.resize(frame, (800, 600)))

            # Press 'q' to stop webcam and return to menu
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nReturned to main menu.")

    elif choice == '2':
        # --- UPLOAD MODE ---
        print("\nOpening file browser...")
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True) # Force pop-up to front
        file_path = filedialog.askopenfilename(
            title="Select Eye Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        root.destroy()

        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                # Predict
                proc_img = preprocess_image(image)
                label, conf = get_prediction(proc_img)

                # Draw results on image
                color = (0, 255, 0) if label == 'Normal' else (0, 0, 255)
                h, w = image.shape[:2]
                scale = 800 / w if w > 800 else 1.0
                display_img = cv2.resize(image, (int(w*scale), int(h*scale)))
                
                # Text background
                cv2.rectangle(display_img, (0, 0), (display_img.shape[1], 60), (0, 0, 0), -1)
                cv2.putText(display_img, f"PREDICTION: {label} ({conf:.1f}%)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                print(f"Prediction Finished: {label} ({conf:.1f}%)")
                cv2.imshow('Upload Result - Press any key to return to menu', display_img)
                cv2.waitKey(0) # Wait until a key is pressed
                cv2.destroyAllWindows()
            else:
                print("Error: Could not read image.")
        else:
            print("No file selected.")

    elif choice == '3':
        print("\nExiting program. Goodbye!")
        break # Exit the while loop

    else:
        print("\nInvalid choice. Please enter 1, 2, or 3.")