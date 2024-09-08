import cv2
import os
import time
import requests

#urls for model and configuration
face_detection_model_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
face_detection_config_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

#local file paths
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/face_rec')
face_detection_model_file = os.path.join(base_dir, 'models', 'deploy.prototxt')
face_detection_config_file = os.path.join(base_dir, 'config_files', 'res10_300x300_ssd_iter_140000.caffemodel')

count = 0

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

#function to create directory with the name of the person taking the pictures
def create_dir():
    global nameID

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    while True:
        name = str(input("enter your name: ")).lower()
        nameID = name

        #create the path in the parent folder for the training and validation dataset
        train_path = os.path.join(parent_dir, 'data/train', name)
        validation_path = os.path.join(parent_dir, 'data/validation', name)

        if os.path.exists(train_path) or os.path.exists(validation_path):
            print("name already taken. please choose a different name.")
        else:
            os.makedirs(train_path)
            os.makedirs(validation_path)
            break

    print(f"directories created at {train_path} and {validation_path}")
    
    #return both paths as a tuple
    return train_path, validation_path

def ask_user_ready():
    while True:
        response = input("are you ready to take the picture? [y/n]: ").strip().lower()
        if response == 'y':
            print("proceeding to take pictures. please, make sure to move and turn your head a little.")
            return True
        elif response == 'n':
            print("exiting the program.")
            return False
        else:
            print("invalid input. please type 'y' or 'n'.")

if not ask_user_ready():
    exit()

#download model and config files
download_file(face_detection_model_url, face_detection_model_file)
download_file(face_detection_config_url, face_detection_config_file)

#create directory and get the path where images will be saved
train_dir, validation_dir = create_dir()

#load the dnn model for face detection
net = cv2.dnn.readNetFromCaffe(face_detection_model_file, face_detection_config_file)

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

            #draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            #extract the face region from the frame
            face_frame = frame[startY:endY, startX:endX]

            #save the image in the correct directory
            if count < 400:
                image_path = os.path.join(train_dir, f'{nameID}.{count+1}.jpg')
                print("creating image........" + image_path)
                cv2.imwrite(image_path, face_frame)
                count += 1
            else:
                image_path = os.path.join(validation_dir, f'{nameID}.{count+1}.jpg')
                print("creating image........" + image_path)
                cv2.imwrite(image_path, face_frame)
                count += 1

    #display the resulting frame
    cv2.imshow('webcam', frame)

    #introduce a short sleep to avoid high cpu usage
    time.sleep(0.01)

    key = cv2.waitKey(1)
    if count == 500:
        print("thanks for the pictures, now finishing the program...")
        cv2.destroyWindow('webcam')
        time.sleep(2)
        break

#release the camera and close all windows
camera.release()
cv2.destroyAllWindows()