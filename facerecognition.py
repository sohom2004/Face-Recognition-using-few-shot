import cv2
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import PIL.ImageOps    
from torch import optim
import torch.nn.functional as F
import os
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
from torchvision import models, transforms, datasets
from datetime import date, datetime
import shutil

# Defining Flask App
app = Flask(__name__)
CORS(app)

port_app = 8000
app.config["MONGO_URI"] = "mongodb://localhost:27017/pep"  # Update with your MongoDB URI
mongo = PyMongo(app)

# Directory paths
main_dir = 'static/faces'
model_save_path = 'static/model_siamese.pth'

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

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

class SiameseDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        # Choose random class and anchor image
        img1, label1 = self.imageFolderDataset.imgs[index]
        label1 = self.imageFolderDataset.targets[index]
        
        # Positive or Negative pair
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            # Get a positive pair (same class)
            while True:
                img2, label2 = random.choice(self.imageFolderDataset.imgs)
                label2 = self.imageFolderDataset.targets[self.imageFolderDataset.imgs.index((img2, label2))]
                if label1 == label2:
                    break
        else:
            # Get a negative pair (different class)
            while True:
                img2, label2 = random.choice(self.imageFolderDataset.imgs)
                label2 = self.imageFolderDataset.targets[self.imageFolderDataset.imgs.index((img2, label2))]
                if label1 != label2:
                    break
        
        # Load the images
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor([int(label1 == label2)], dtype=torch.float32)  # 1 for similar, 0 for dissimilar

    def __len__(self):
        return len(self.imageFolderDataset)

# Define transformations for the training data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data as pairs
image_folder = datasets.ImageFolder(root=main_dir)
siamese_dataset = SiameseDataset(image_folder, transform=transform)
train_loader = DataLoader(siamese_dataset, batch_size=16, shuffle=True)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, len(image_folder.classes))  # Embedding size

    def forward_once(self, x):
        # Forward pass through the network backbone
        return self.backbone(x)

    def forward(self, input1, input2):
        # Forward both images through the same network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) + 
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
    
if os.path.exists(model_save_path):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()
else:
    model = SiameseNetwork()


def train_siamese_model():
    
    # Define loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for img1, img2, label in train_loader:
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    

def load_model():
    """Load the trained Siamese network model and adjust the state_dict."""
    model = SiameseNetwork()
    
    # Load the saved state_dict
    state_dict = torch.load(model_save_path, map_location=torch.device('cpu'))
    
    # Adjust the state_dict to remove 'backbone.' prefix
    adjusted_state_dict = {key.replace("backbone.", ""): value for key, value in state_dict.items()}
    
    # Load the adjusted state_dict into the model
    model.backbone.load_state_dict(adjusted_state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

def add_attendance(name):
    pass


from torch.nn.functional import cosine_similarity

def verify_identity(input_face_tensor, reference_face_tensor, model, threshold=0.60):
    """Verify the identity by comparing the input face with a reference face using Siamese Network."""
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        # Pass both faces through the Siamese Network
        output1, output2 = model(input_face_tensor, reference_face_tensor)
        
        # Calculate similarity (e.g., cosine similarity)
        similarity = cosine_similarity(output1, output2).item()
    
    # If similarity exceeds the threshold, we assume it's the same identity
    if similarity > threshold:
        return True, similarity
    else:
        return False, similarity


def identify_face(input_face, model):
    """Identify the face using the trained Siamese model."""
    
    # Transformation for the input face
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert the input face into a tensor
    input_face = Image.fromarray(cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB))  # Convert to PIL image
    input_face_tensor = transform(input_face).unsqueeze(0)  # Apply transformations and add batch dimension
    
    # Initialize variables for storing the best match
    best_match = None
    highest_similarity = 0.0
    
    # Iterate over known face directories in the 'static/faces' directory
    for class_name in os.listdir(main_dir):
        class_dir = os.path.join(main_dir, class_name)
        
        # Ensure it's a directory (folder containing images for this person)
        if os.path.isdir(class_dir):
            # Iterate over images within the folder
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)

                # Check if the path is a file (not a directory)
                if not os.path.isfile(img_path):
                    continue

                try:
                    # Load the reference face image
                    reference_face = plt.imread(img_path)
                    reference_face_pil = Image.fromarray(reference_face)
                    reference_face_tensor = transform(reference_face_pil).unsqueeze(0)

                    # Compare the input face with the reference face using the Siamese network
                    match, similarity = verify_identity(input_face_tensor, reference_face_tensor, model)

                    # Keep track of the best match
                    if match and similarity > highest_similarity:
                        best_match = class_name
                        highest_similarity = similarity

                except PermissionError:
                    print(f"Permission denied for file: {img_path}")
                    continue  # Skip files with permission issues

    # If a match is found, return the best match with the similarity score
    if best_match:
        return f'{best_match} ({highest_similarity:.2f})'
    else:
        return "Unknown"
    
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

    if os.listdir('static/faces/') == []:
        os.remove('static/model.pth')
    
    try:
        train_siamese_model()
    except:
        pass
    return jsonify({})

# Face Recognition and Attendance Route
@app.route('/start/', methods=['GET', 'POST'])
def start():
    print("Received request to take attendance")

    # Check if the trained model exists
    if 'model_siamese.pth' not in os.listdir('static'):
        return jsonify({
            "message": 'There is no trained model in the static folder. Please add a new face to continue.'
        }), 400

    # Load the trained Siamese model
    model = load_model()

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"message": "Webcam not accessible"}), 500

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Detect face bounding boxes
        faces = detect_bounding_box(frame)
        if len(faces) > 0:
            # Only process the first detected face (assuming single person detection)
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)

            # Extract the face from the frame for identification
            face = frame[y:y+h, x:x+w]
            
            # Identify the person using the Siamese network
            identified_person = identify_face(face, model)

            # If a person is identified, add them to attendance and display on frame
            if identified_person != "Unknown":
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame with the bounding box and label
        cv2.imshow('Attendance', frame)
        
        # Press 'Esc' key to exit
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the 'Esc' key
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Attendance process completed"})
    
@app.route('/add/', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    # Make sure this folder exists
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"message": "Webcam not accessible"}), 500

    capturing = True  # in case of capture key implementation if needed

    while True:
        _, frame = cap.read()
        faces = detect_bounding_box(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if capturing and j % 5 == 0 and i < 5:
                name = f'new_{i}.jpg'  # Save each image with a different name
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                i += 1  # Increment image count
            j += 1
        cv2.putText(frame, f'Images Captured: {i}/5', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Adding new User', frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) == 27:  # Escape key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_siamese_model()  # Call the training function after capturing images
    return jsonify({})


# Run Flask App
if __name__ == '__main__':
    print(f"Python server is running at {port_app}")
    app.run(debug=True, port=port_app)