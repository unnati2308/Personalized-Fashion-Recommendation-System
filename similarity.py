import pickle
import tensorflow as tf
import numpy as np
import shutil
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import os

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

def find_similar_images(sample_image_path):
    img = image.load_img(sample_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([normalized_result])

    output_dir = 'static/similar_images'
    os.makedirs(output_dir, exist_ok=True)

    similar_image_paths = []
    for idx in indices[0][1:6]:
        source_path = filenames[idx]
        destination_path = os.path.join(output_dir, os.path.basename(source_path))
        shutil.copyfile(source_path, destination_path)
        similar_image_paths.append(f'similar_images/{os.path.basename(source_path)}')

    return similar_image_paths
