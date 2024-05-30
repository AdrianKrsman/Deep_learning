# Import packages
import threading 
import os 
import time

import cv2
import numpy as np
from deepface import DeepFace
from keras.models import load_model
from keras.utils import img_to_array



# Initialize web camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the size of the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize counter and default values for face_match and emotion
counter = 0
face_match = False
emotion = "Detecting"

# Create empty list for reference images to be added
reference_imgs = []
# Save reference images in directory
reference_dir = "reference_images"

# Load face detection using Harr Cascade model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the emotion classifier model and define emotion labels
mood_classifier = load_model("emotion_detector.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Create a lock for thread-safe operations
lock = threading.Lock()

# Create function to save reference images
def save_reference_images():
    if not os.path.exists(reference_dir):
        os.makedirs(reference_dir)
    for idx, img in enumerate(reference_imgs):
        filename = os.path.join(reference_dir, f"reference_{idx}.png")
        cv2.imwrite(filename, img)
    print("Reference images saved to disk.")

# Create function to load reference images
def load_reference_images():
    global reference_imgs
    reference_imgs = []
    if os.path.exists(reference_dir):
        for filename in os.listdir(reference_dir):
            img_path = os.path.join(reference_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                reference_imgs.append(img)
    print("Reference images loaded from disk.")

# Load reference images
load_reference_images()

# Create function to add current frame to reference images
def save_frame_and_add_reference():
    ret, frame = cap.read()
    if ret:
        global reference_imgs
        reference_imgs.append(frame)
        save_reference_images()
        print("Reference image added and saved to disk.")

# Function to verify if face matches face in reference images and classify emotion 
def check_face(face_region):
    global face_match
    try:
        # Verify if the detected face matches reference images
        verified = any(DeepFace.verify(face_region, ref_img.copy())['verified'] for ref_img in reference_imgs)
        with lock:
            face_match = verified

    except ValueError as e:
        with lock:
            face_match = False
            emotion = 'Unknown'
            print(f"Error during prediction: {e}")

# Create function to detect emotion on face
def check_emotion(face_region):
    global emotion
    try:
        # Preprocess detected face for emotion prediction
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray_face,(48,48))
        face = face.astype('float')/255.0
        face = img_to_array(face)
        face = np.expand_dims(face,axis=0)

        # Predict emotion of the detected face
        emotion_prediction = mood_classifier.predict(face)[0]
        emotion_label = emotion_labels[emotion_prediction.argmax()] 
        with lock:
            emotion = emotion_label

    except ValueError as e:
        with lock:
            emotion = 'Unknown'
            print(f"Error during prediction: {e}")

while True:
    ret, frame = cap.read()

    if ret:
        # Convert to grayscale and detect faces in frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.1,
                                              minNeighbors = 5, minSize = (30, 30))
        
        # Iterate through detected faces and extract the frame of the face
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            # Check face and emotion every 30 frames
            if counter % 30 == 0:
                threading.Thread(target = check_face, args=(face_region,)).start()
                threading.Thread(target = check_emotion, args = (face_region,)).start()
            with lock:
                # Display face match result
                if face_match:
                    cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                # Deny access if emotion is angry or fearful
                if face_match and (emotion == 'Angry' or emotion == 'Fear'):
                    cv2.putText(frame, "ACCESS DENIED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(frame, "Try again later!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.imshow("video", frame)
                    cv2.waitKey(1)
                    time.sleep(3)
                    counter = 0
                    continue
                
                # Display the detected emotion in frame
                cv2.putText(frame, f"{emotion}", (370, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                
            cv2.imshow("video", frame)
            counter += 1

    key = cv2.waitKey(1)
    # Break if "q" is pressed
    if key == ord("q"):
        break
    # Save frame to reference images using created function if "s" is pressed
    elif key == ord("s"):
        save_frame_and_add_reference()

cap.release()
cv2.destroyAllWindows()