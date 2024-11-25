import cv2
import numpy as np
import pandas as pd
import dlib
import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
from datetime import date, datetime
import shutil
from scipy.spatial import distance

# Defining Flask App
app = Flask(__name__)
CORS(app)

port_app = 8000
app.config["MONGO_URI"] = "mongodb://localhost:27017/pep"  # Update with your MongoDB URI
mongo = PyMongo(app)

# Initialize dlib's face detector and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory for known faces
known_faces_dir = "static/faces/"
known_embeddings = []
known_labels = []

current_date = datetime.now().strftime("%Y%m%d")
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')
        
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_bounding_box(frame):
    """Detect faces in the given frame using OpenCV's Haar cascades or dlib."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Load known faces and their embeddings
def load_known_faces():
    global known_embeddings, known_labels
    known_embeddings.clear()
    known_labels.clear()
    for class_name in os.listdir(known_faces_dir):
        class_path = os.path.join(known_faces_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    detections = face_detector(rgb_img, 1)

                    if len(detections) == 1:  # Process only if one face is detected
                        shape = shape_predictor(rgb_img, detections[0])
                        embedding = np.array(face_rec_model.compute_face_descriptor(rgb_img, shape))
                        known_embeddings.append(embedding)
                        known_labels.append(class_name)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
# Recognize face in the input frame
def recognize_face(rgb_img, face_rect):
    shape = shape_predictor(rgb_img, face_rect)
    embedding = np.array(face_rec_model.compute_face_descriptor(rgb_img, shape))

    # Compare with known embeddings
    min_distance = float("inf")
    best_match = "Unidentified"
    for known_emb, label in zip(known_embeddings, known_labels):
        dist = distance.euclidean(embedding, known_emb)
        if dist < min_distance:
            min_distance = dist
            best_match = label

    # Return match if within threshold
    threshold = 0.6  # Adjust threshold as needed
    if min_distance < threshold:
        return best_match, min_distance
    else:
        return "Unidentified", min_distance

# delete a folder    
def deletefolder(folder_path):
    """Recursively delete a folder and its contents."""
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder {folder_path} has been deleted successfully.")
        else:
            print(f"Folder {folder_path} does not exist.")
    except Exception as e:
        print(f"Error deleting folder {folder_path}: {str(e)}")
        
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    filter = {
        "domain": g_domain,
        "batch": int(g_batch),
        "students.regisno": int(userid),
    }
    
    update = {
        "$set": {
             "students.$.afterPresent": True
         }
     }
    
    result = mongo.db.mains.update_one(filter, update)
    
    if result.matched_count > 0:
        print("Student updated successfully")
    else:
        print("Student not found")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
            

    
################## ROUTING FUNCTIONS #########################
g_domain = "FullStack"
g_batch = 1

@app.route('/')
def home():
    if not os.path.isdir(f'static/faces/{g_domain}'): 
        os.makedirs(f'static/faces/{g_domain}')
    if not os.path.isdir(f'static/faces/{g_domain}/{g_batch}'): 
        os.makedirs(f'static/faces/{g_batch}')
    return jsonify({})

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder(f'static/faces/{duser}')
    return jsonify({})

# Face Recognition and Attendance Route
@app.route('/start/', methods=['GET', 'POST'])
def start():
    print("Received request to start real-time recognition")

    # Ensure known faces are loaded
    load_known_faces()
    if len(known_embeddings) == 0:
        return jsonify({"message": "No known faces loaded. Add known faces to the static/faces directory."}), 400

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"message": "Webcam not accessible"}), 500

    identified_person = None  # To store the identified person's name

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(rgb_frame, 0)

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            name, dist = recognize_face(rgb_frame, face)

            # Draw bounding box and label
            color = (0, 255, 0) if name != "Unidentified" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({dist:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if name != "Unidentified":
                identified_person = name
                print(f"Identified: {name} (Distance: {dist:.2f})")
                break

        # Display the frame
        cv2.imshow("Attendance", frame)

        # If a person is identified, exit the loop
        if identified_person:
            break

        # Press 'q' to quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    if identified_person:
        return jsonify({"message": f"Recognition complete: {identified_person}"})
    else:
        return jsonify({"message": "No match found. Attendance not recorded."})

    
@app.route('/add/', methods=['POST'])
def add():
    # Retrieve username and ID from the request
    newusername = request.form.get('newusername')
    newuserid = request.form.get('newuserid')

    if not newusername or not newuserid:
        return jsonify({"message": "Missing username or user ID"}), 400

    # Create a folder for the new user
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)

    # Initialize variables
    padding_ratio = 0.3  # Adjust the cropping size (30% padding)
    max_images = 5       # Maximum number of images to capture
    image_count = 0      # Counter for the number of images saved

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"message": "Webcam not accessible"}), 500

    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Detect faces in the frame
        faces = detect_bounding_box(frame)

        for (x, y, w, h) in faces:
            if image_count >= max_images:
                break

            # Calculate padding
            padding_w = int(w * padding_ratio)
            padding_h = int(h * padding_ratio)

            # Adjust crop dimensions with padding
            x_new = max(0, x - padding_w)
            y_new = max(0, y - padding_h)
            w_new = w + 2 * padding_w
            h_new = h + 2 * padding_h

            # Save cropped image
            name = f'new_{image_count + 1}.jpg'
            cropped_face = frame[y_new:y_new+h_new, x_new:x_new+w_new]
            cv2.imwrite(os.path.join(userimagefolder, name), cropped_face)
            image_count += 1

            # Draw rectangle for visualization
            cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 0, 20), 2)

        # Display feedback on the frame
        cv2.putText(frame, f'Images Captured: {image_count}/{max_images}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Adding new User', frame)

        # Automatically stop if max images are captured
        if image_count >= max_images:
            break

        # Break loop if the 'Esc' key is pressed during the process
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for 'Esc'
            break

    # Release webcam and close the display window
    cap.release()
    cv2.destroyAllWindows()

    # Return response
    if image_count == max_images:
        print(f"Successfully captured {max_images} images for {newusername}_{newuserid}")
        return jsonify({"message": f"Successfully added {newusername}_{newuserid} with {max_images} images."})
    else:
        print(f"Only captured {image_count} images.")
        return jsonify({"message": f"Captured only {image_count} images. Please try again."}), 400

# Run Flask App
if __name__ == '__main__':
    print(f"Python server is running at {port_app}")
    app.run(debug=True, port=port_app)