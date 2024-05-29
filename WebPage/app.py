from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='templates/static')
CORS(app)  # Enable CORS

# Load your pre-trained Keras model
model = load_model(r"C:\Users\havar\Downloads\Project\DeepLearning_model_output\Trained Models\MobileNetV2\MobileNetV2_model.keras")  # Corrected file path

# Define class labels
class_labels = ['Alstonia scholaris', 'Citrus aurantiifolia', 'Jatropha', 'Mangifera indica', 'Ocimum basilicum', 'Platanus orientalis', 'Pomegranate', 'Pongamia Pinnata', 'Psidium guajava', 'Syzygium cumini', 'Terminalia arjuna']

def preprocess_image(image):
    # Convert the image to RGB if it has an alpha channel (RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to fit the model input requirements
    image = image.resize((224, 224))  # Adjust size based on your model's input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Predict using your model
        prediction = model.predict(processed_image)[0]
        
        # Get top 5 predictions
        top_indices = prediction.argsort()[-5:][::-1]
        top_predictions = [(class_labels[i], float(prediction[i])) for i in top_indices]
        
        # Return the prediction as JSON
        return jsonify({'predictions': top_predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # Runs the server locally on port 5000
