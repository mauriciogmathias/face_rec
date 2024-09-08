import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

#set up directories with relative paths
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/face_rec')
models_dir = os.path.join(base_dir, 'models')
labels_file_path = os.path.join(base_dir, 'labels', 'labels.txt')
model_path = os.path.join(models_dir, "InceptionV3_transfer_learning.keras")

#load the saved model
print("loading model from:", model_path)
model = tf.keras.models.load_model(model_path)

#print the model summary
model.summary()

#function to load class names from labels.txt
def load_class_names(labels_file_path):
    class_names = []
    with open(labels_file_path, 'r') as file:
        for line in file:
            index, class_name = line.strip().split(' ')
            class_names.append(class_name)
    return class_names

#load class names from the labels.txt file
class_names = load_class_names(labels_file_path)
print(f"class names loaded from {labels_file_path}: {class_names}")

#list of test images 
test_image_paths = [
    '/home/mauricio/Desktop/ml/face_rec/data/validation/mauricio/mauricio.494.jpg',
    '/home/mauricio/Desktop/ml/face_rec/data/validation/ana/ana.473.jpg',
    '/home/mauricio/Desktop/ml/face_rec/data/validation/carmen/carmen.499.jpg', 
]

#loop through each test image
for test_image_path in test_image_paths:
    if os.path.exists(test_image_path):
        print(f"testing image: {test_image_path}")
        
        #load and preprocess the image
        img = image.load_img(test_image_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        #preprocess the image to match the input requirements of inceptionv3
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        #make predictions
        predictions = model.predict(img_array)
        
        #print raw prediction probabilities for all classes
        print(f"raw predictions for {test_image_path}: {predictions}")

        #get the predicted class index (class with the highest probability)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        print(f"predicted class: {predicted_label} with probability {predictions[0][predicted_class]}")

        #show the top 3 predicted classes with their probabilities
        top_3 = np.argsort(predictions[0])[-3:][::-1]
        for i in top_3:
            print(f"class {i} ({class_names[i]}) with probability {predictions[0][i]}")
    else:
        print(f"file not found: {test_image_path}")
