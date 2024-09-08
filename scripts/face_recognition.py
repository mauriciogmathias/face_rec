import cv2
import os
import time
import requests
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

#urls for model and configuration
face_detection_model_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
face_detection_config_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

#local file paths
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/face_rec')
models_dir = os.path.join(base_dir, 'models')
labels_file_path = os.path.join(base_dir, 'labels', 'labels.txt')
model_path = os.path.join(models_dir, "InceptionV3_transfer_learning.keras")
face_detection_model_file = os.path.join(base_dir, 'models', 'deploy.prototxt')
face_detection_config_file = os.path.join(base_dir, 'config_files', 'res10_300x300_ssd_iter_140000.caffemodel')

#function to download model files if they don't exist
def download_file(url, file_name):
    if not os.path.exists(file_name):
        print(f"downloading {file_name}...")
        response = requests.get(url)
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} downloaded successfully.")
    else:
        print(f"{file_name} already exists, skipping download.")

#download model and config files
download_file(face_detection_model_url, face_detection_model_file)
download_file(face_detection_config_url, face_detection_config_file)

#load the dnn model for face detection
print("loading facial detection model from:", face_detection_model_file)
net = cv2.dnn.readNetFromCaffe(face_detection_model_file, face_detection_config_file)

#load the saved model
print("loading facial recognition model from:", model_path)
model = tf.keras.models.load_model(model_path)

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

#initialize camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("can't open camera")
    exit()

#set camera resolution
desired_width = 1920
desired_height = 1080
camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

#make the opencv window resizable
cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('webcam', desired_width, desired_height)

while True:
    #capture frame-by-frame
    ret, frame = camera.read()

    if not ret:
        print("can't receive frame, exiting...")
        break

    h, w = frame.shape[:2]

    #prepare the frame for dnn model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    #loop over the detections and recognize faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        #filter out weak detections by confidence
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            #extract the face region from the frame
            face_frame = frame[startY:endY, startX:endX]

            #resize the face region to 160x160
            face_frame_resized = cv2.resize(face_frame, (160, 160))
            img_array = image.img_to_array(face_frame_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            #preprocess the image to match the input requirements of inceptionv3
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

            #make predictions
            predictions = model.predict(img_array)

            #get the predicted class index (class with the highest probability)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]

            if predictions[0][predicted_class] > 0.9:
                name = predicted_label
                accuracy_text = f"{predictions[0][predicted_class] * 100:.2f}%"
                cv2.putText(frame, accuracy_text, (startX, endY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                name = "unknown"

            print(f"predicted class: {predicted_label} with probability {predictions[0][predicted_class]}")
            
            #draw the bounding box and the name
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #display the resulting frame
    cv2.imshow('webcam', frame)

    #introduce a short sleep to avoid high cpu usage
    time.sleep(0.01)

    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty('webcam', cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()