import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image

# Directory paths
main_dir = 'static/faces'
model_save_path = 'static/siamese_model.pth'

# Custom dataset to generate pairs of images
class SiameseDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.classes = image_folder_dataset.classes

    def __getitem__(self, index):
        # Randomly choose whether the pair is "same" or "different"
        should_get_same_class = random.randint(0, 1)

        # Get an image and its class label
        img1, label1 = self.image_folder_dataset[index]

        if should_get_same_class:
            # Get another image from the same class
            while True:
                img2, label2 = random.choice(self.image_folder_dataset.samples)
                if label1 == label2:
                    break
        else:
            # Get an image from a different class
            while True:
                img2, label2 = random.choice(self.image_folder_dataset.samples)
                if label1 != label2:
                    break

        # Apply transformations if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # The label is 0 for same class and 1 for different class
        label = torch.tensor([int(label1 != label2)], dtype=torch.float32)

        return img1, img2, label

    def __len__(self):
        return len(self.image_folder_dataset)

# Siamese Network using a pretrained ResNet18
class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()  # Remove the final classification layer
        self.resnet = resnet
        self.fc = nn.Linear(512, 1)  # Output one value for similarity

    def forward_once(self, x):
        # Pass the input through ResNet to get the embeddings
        return self.resnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Compute the L1 distance between the two embeddings
        distance = torch.abs(output1 - output2)
        output = self.fc(distance)
        return output

# Contrastive loss for training
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # Compute the contrastive loss
        loss = (1 - label) * 0.5 * torch.pow(output, 2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - output, min=0), 2)
        return loss.mean()

# Model Training Function for One-Shot Learning
def train_siamese_network(epochs=10):
    # Define transformations for the input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the data
    image_folder_dataset = datasets.ImageFolder(root=main_dir)
    siamese_dataset = SiameseDataset(image_folder_dataset, transform=transform)
    train_loader = DataLoader(siamese_dataset, shuffle=True, batch_size=32)

    # Initialize the model
    model = SiameseResNet()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for img1, img2, label in train_loader:
            optimizer.zero_grad()
            # Forward pass
            output = model(img1, img2)
            loss = criterion(output, label)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
