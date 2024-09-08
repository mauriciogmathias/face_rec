Face Recognition System - README
=================================

Project Overview
----------------
This project aims to create a simple but effective face recognition system. It includes:
1. Data Collection: Capture pictures of different faces to create a training dataset.
2. Model Training: Train a deep learning model using TensorFlow and InceptionV3 for face recognition.
3. Testing: Evaluate the trained model on new face images.
4. Real-Time Recognition: Use a webcam to detect and recognize faces in real-time.

Structure
---------
Here’s how the project is structured:
- `data/`: Contains training and validation datasets, organized by person.
- `models/`: Stores the trained models and detection files.
- `config_files/`: Contains the configuration and model files for face detection.
- `labels/`: Stores class labels of the people recognized.
- `images/`: Contains example images and results.
- `logs/`: Training logs for TensorBoard.
- `scripts/`: Holds all the Python scripts for data collection, training, testing, and recognition.

How to Run
----------
1. **Data Collection (Step 1)**
   First, capture images of the person you want the model to recognize:
   - Run the `data_collection.py` script located in the `scripts/` directory.
   - It will ask for your name and create directories for you.
   - The system will take 500 pictures (400 for training, 100 for validation).
   
   **Tip:** Move your head a bit for better training data!

2. **Model Training (Step 2)**
   Once the images are captured, you can train the model.
   - Run `train_model.py` from the `scripts/` directory to start the training process.
   - This script will:
     - Use InceptionV3 as a base model.
     - Fine-tune it with your collected dataset.
     - Save the best version of the model in the `models/` directory.
   - After training, it will output graphs showing the training/validation accuracy and loss.

3. **Testing (Step 3)**
   Test the model on new images:
   - Run `test_model.py` located in the `scripts/` directory to check if the model recognizes new images.
   - You’ll see the predicted class (person’s name) and probabilities.

4. **Real-Time Recognition (Step 4)**
   Finally, run the face recognition system using your webcam:
   - Launch the `face_recognition.py` script from the `scripts/` directory.
   - The system will detect faces in real-time using OpenCV.
   - If a face is recognized with high confidence (90% or more), the person's name will appear, otherwise "Unknown."

Requirements
------------
You'll need the following libraries installed:
- **OpenCV**: For handling images and webcam input.
- **TensorFlow**: For training and recognizing faces.
- **Matplotlib**: For visualizing data during training.
- **Requests**: To download model files.