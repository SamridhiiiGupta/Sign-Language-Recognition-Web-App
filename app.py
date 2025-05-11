from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your trained model from the "model" folder
model = load_model('model/sign_language_model.h5')

# Home route — shows the interface (HTML page)
@app.route('/')
def index():
    return render_template('index.html')

# This route will handle the image sent by JavaScript and return a prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read the uploaded image from the form
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')
    image = image.resize((64, 64))  # Resize to match model input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': str(predicted_class)})

if __name__ == '__main__':
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")

    # Start a timer to open the browser after 1 second
    Timer(1, open_browser).start()

    app.run(debug=True)

