import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

import pandas as pd
import shutil

csv_path = 'C:/Users/Public/Downloads/DLLABPROJECT/styles.csv'  # Update with the actual path to styles.csv
images_dir = 'C:/Users/Public/Downloads/DLLABPROJECT/images'  # Update with the path where images are stored
output_dir = 'C:/Users/Public/Downloads/DLLABPROJECT/cleaned_images'  # Directory to store only "Apparel" images
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(csv_path, on_bad_lines='skip')

dress_df = df[df['masterCategory'] == 'Apparel']

for _, row in dress_df.iterrows():
    img_filename = f"{row['id']}.jpg"  # Assuming image filenames match the 'id' column in styles.csv
    img_path = os.path.join(images_dir, img_filename)


    if os.path.exists(img_path):
        shutil.copy(img_path, output_dir)
    else:
        print(f"Warning: {img_filename} not found in {images_dir}")

print("Filtering complete. Only Apparel images are now in", output_dir)

num_images = len([file for file in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, file))])
print("Number of images in cleaned_images: ", num_images)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

cleaned_images_dir = 'C:/Users/Public/Downloads/DLLABPROJECT/cleaned_images'

filenames = [os.path.join(cleaned_images_dir, file) for file in os.listdir(cleaned_images_dir) if os.path.isfile(os.path.join(cleaned_images_dir, file))]

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("Feature extraction complete. Embeddings saved to 'embeddings.pkl'")
