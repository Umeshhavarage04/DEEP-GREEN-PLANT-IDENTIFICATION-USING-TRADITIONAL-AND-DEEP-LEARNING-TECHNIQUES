import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model(r"C:\Users\karanbeer\Desktop\VSCODE\FINAL PROJECT\Trained Models\MobileNetV2\MobileNetV2_model.keras")

# Constants
IMAGE_RES = 224

# Define the path to the dataset directory
dataset_dir = r"C:\Users\karanbeer\Desktop\VSCODE\FINAL PROJECT\augmented_images"

# Extract class names from subdirectories in the dataset directory
class_names = sorted(os.listdir(dataset_dir))

# Load and preprocess the input image
input_image_path = r"D:\download.jpg"
input_image = load_img(input_image_path, target_size=(IMAGE_RES, IMAGE_RES))
input_image_array = img_to_array(input_image)
input_image_array = input_image_array / 255.0  # Normalize the image
input_image_array = input_image_array[tf.newaxis, ...]

# Make predictions using the loaded model
predictions = model.predict(input_image_array)[0]

# Get the top 5 predictions
top_5_indices = predictions.argsort()[-5:][::-1]
top_5_predictions = [(class_names[i], predictions[i] * 100) for i in top_5_indices]

# Display the input image along with the top 5 predicted class names and confidence scores
plt.imshow(input_image)
plt.title("Top 5 Predictions:")
plt.axis("off")

# Annotate the image with the top 5 predictions
for i, (class_name, confidence) in enumerate(top_5_predictions):
    plt.text(10, 30 + i * 20, f"{i+1}. {class_name}: {confidence:.2f}%", color='white', fontsize=12, backgroundcolor='black')

plt.show()
