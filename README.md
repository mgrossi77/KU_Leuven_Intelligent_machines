# KU_Leuven_Intelligent_machines
#Intelligent Machines course
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load a pre-trained CNN model (MobileNetV2 trained on ImageNet)
model = MobileNetV2(weights="imagenet")

# Load and preprocess image
image = cv2.imread("image.jpg")
image = cv2.resize(image, (224, 224))  # MobileNetV2 expects 224x224 images
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Normalize image (MobileNetV2 expects pixel values between -1 and 1)
image = image / 127.5 - 1

# Predict object category
predictions = model.predict(image)

# Decode the predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Print the top-3 predicted labels
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} ({score:.2f})")
