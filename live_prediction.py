import cv2
import numpy as np
from keras.models import load_model
import pickle

# Load the trained model
model = load_model('model/sign_language_model.h5')

# Load the label encoder
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the width and height of the webcam window
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is not captured, break the loop
    if not ret:
        break
    
    # Preprocess the frame (resize and normalize)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))  # Resize to 64x64 pixels
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image)
    
    # Decode the prediction
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    
    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {predicted_label[0]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame with the prediction
    cv2.imshow('Sign Language Recognition', frame)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()

