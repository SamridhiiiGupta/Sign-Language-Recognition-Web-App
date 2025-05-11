import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(data_path, image_size=(64, 64)):
    data = []
    labels = []
    class_names = os.listdir(data_path)

    for label in class_names:
        class_folder = os.path.join(data_path, label)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize
                    data.append(img)
                    labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    return data, labels_categorical, le
