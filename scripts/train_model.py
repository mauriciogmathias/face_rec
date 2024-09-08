import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

#set tensorflow threading options for cpu optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

#set up directories
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/face_rec')
train_dir = os.path.join(base_dir, 'data/train')
validation_dir = os.path.join(base_dir, 'data/validation')
output_dir = os.path.join(base_dir, 'images')
models_dir = os.path.join(base_dir, 'models')
logs_dir = os.path.join(base_dir, 'logs')
labels_dir = os.path.join(base_dir, 'labels')
labels_file_path = os.path.join(labels_dir, 'labels.txt')

#preprocess both the training and validation datasets for inceptionv3
def preprocess_for_inception(image, label):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

#create directories if they don't exist
for dir in [output_dir, models_dir, logs_dir, labels_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

print("tensorflow version:", tf.__version__)

#training parameters
batch_size = 16
img_size = (160, 160)

#load datasets with validation split
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

#save class names to labels.txt
class_names = train_dataset.class_names
print("class names:", class_names)
with open(labels_file_path, 'w') as f:
    for index, class_name in enumerate(class_names):
        f.write(f'{index} {class_name}\n')

#visualize and save a few training samples
for images, labels in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.savefig(os.path.join(output_dir, 'train_sample_images.png'))
    plt.close()

train_dataset = train_dataset.map(preprocess_for_inception)
validation_dataset = validation_dataset.map(preprocess_for_inception)

#cache dataset and prefetch for better performance
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)
])

#apply data augmentation to the training dataset after preprocessing
augmented_train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

#check class distribution
class_counts = {}
for _, labels in train_dataset.unbatch():
    label = labels.numpy()
    if label not in class_counts:
        class_counts[label] = 0
    class_counts[label] += 1
print("class distribution in training set:", class_counts)

#load inceptionv3, without the top classification layer
img_shape = img_size + (3,)
base_model = tf.keras.applications.InceptionV3(input_shape=img_shape, include_top=False, weights='imagenet')
base_model.trainable = False

#add new layers on top of inceptionv3 base
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
num_classes = len(class_names)
prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.inception_v3.preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.6)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

#unfreeze part of the base model, fine-tune only the last layers of the inceptionv3 model
#explanation: unfreezing from layer 249 allows us to fine-tune the more specialized layers closer to the output,
#while keeping the early layers (which capture general features like edges and textures) frozen.
#this strategy helps the model adapt to our specific dataset without overfitting,
#leveraging the pre-trained general features and focusing training on task-specific layers.
fine_tune_at = 249 

base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

#compile the model with a lower learning rate for fine-tuning
base_learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#class weights calculation
class_indices = []
for _, labels in augmented_train_dataset.unbatch():
    class_indices.append(labels.numpy())
class_indices = np.array(class_indices)

#compute class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(class_indices),
                                     y=class_indices)
class_weights = dict(enumerate(class_weights))
print("computed class weights:", class_weights)

#callbacks
model_save_path = os.path.join(models_dir, "InceptionV3_transfer_learning.keras")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

#training the model
initial_epochs = 30
history = model.fit(augmented_train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    class_weight=class_weights,
                    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

#plot and save metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.ylim([0, 1])
plt.title('training and validation accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.legend(loc='upper right')
plt.ylabel('cross entropy')
plt.ylim([0, 2])
plt.title('training and validation loss')
plt.xlabel('epoch')

plt.savefig(os.path.join(output_dir, 'training_validation_metrics.png'))
plt.close()