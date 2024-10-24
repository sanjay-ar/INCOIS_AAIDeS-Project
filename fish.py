import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from twilio.rest import Client

# Load the YOLOv8 model (assume that the model is stored at 'best.pt')
yolo_model = YOLO(r"/Users/sanjayar/Desktop/incois/best.pt")

# Load the ResNet50 model (assume that the model is stored at 'resnet50_model.h5')
resnet_model = load_model(r'/Users/sanjayar/Desktop/incois/fish_model.h5')

# Twilio credentials
account_sid = 'AC4071b49d7a8f1fdbcf95ca37a3c2a6f4'
auth_token = 'e361ce8abc96b034aec7516f01beec31'
twilio_number = 'whatsapp:+14155238886'
recipient_number = 'whatsapp:+919597176002'

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Function to send a WhatsApp message
def send_whatsapp_message(body):
    message = client.messages.create(
        body=body,
        from_=twilio_number,
        to=recipient_number
    )
    print(f"Message sent: {message.sid}")

# Function to preprocess the image for ResNet50
def preprocess_image_for_resnet(image):
    # Resize the image to the correct input size for ResNet50 (256x256)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict species using the ResNet50 model
def predict_species(cropped_image, resnet_model):
    preprocessed_image = preprocess_image_for_resnet(cropped_image)
    prediction = resnet_model.predict(preprocessed_image)
    species_index = np.argmax(prediction)
    return species_index

# Function to run inference using YOLOv8 and ResNet50 models
def run_inference(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load image.")

    # Run YOLOv8 model to detect objects
    results = yolo_model(image)

    # Extract the bounding box of the detected objects
    if len(results) > 0:
        species_names = {0: 'Black Sea Sprat', 1: 'Black Sea Sprat GT', 2: 'Gilt-Head Bream', 3: 'Gilt-Head Bream GT', 4: 'Hourse Mackerel', 5: 'Hourse Mackerel GT', 6: 'Red Mullet', 7: 'Red Mullet GT', 8: 'Red Sea Bream', 9: 'Red Sea Bream GT', 10: 'Sea Bass', 11: 'Sea Bass GT', 12: 'Shrimp', 13: 'Shrimp GT', 14: 'Striped Red Mullet', 15: 'Striped Red Mullet GT', 16: 'Trout', 17: 'Trout GT'}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cropped_image = image[y1:y2, x1:x2]

                # Predict species using ResNet50 model
                species_index = predict_species(cropped_image, resnet_model)
                species_name = species_names.get(species_index, "Unknown species")
                print(f"Predicted Species Index: {species_index}")
                print(f"Predicted Species Name: {species_name}")

                # Check if species index is 5 or 10, and send a WhatsApp message
                if species_index in [5, 10]:
                    send_whatsapp_message(f"Alert: {species_name} detected in the image!")
    else:
        print("No objects detected.")

# Example usage
image_path = r"/Users/sanjayar/Desktop/incois/fish.jpeg"
run_inference(image_path)
