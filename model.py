import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle

print(tf.__version__)
print('TensorFlow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

# Setup GPU for model run
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Identify number of classes in the dataset
classes = pd.read_csv("data/Train.csv")
classes_no = len(classes.ClassId.unique())

# Define data paths
cwd = os.getcwd()
base_dir = os.path.join(cwd, 'data')
train_path = os.path.join(base_dir, 'Train')
test_path = os.path.join(base_dir, 'Test')
model_path = os.path.join(cwd, "trained_model/lenet_acc98")

# Define constants
BATCH_SIZE = 150
STEPS_PER_EPOCH = 2000
TARGET_SIZE = (32, 32)

# Create a data generator for the training images
train_datagen = ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                validation_split=0.2)  # val 20%

# Create a data generator for the validation images
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Split data to training and validation datasets
train_data = train_datagen.flow_from_directory(train_path,
                                               target_size=TARGET_SIZE,
                                               color_mode='grayscale',
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=2,
                                               subset='training')

val_data = val_datagen.flow_from_directory(train_path,
                                           target_size=TARGET_SIZE,
                                           color_mode='grayscale',
                                           batch_size=BATCH_SIZE,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=2,
                                           subset='validation')

datagen = ImageDataGenerator(rescale=1./255)
test_data = datagen.flow_from_directory(test_path,
                                        target_size=TARGET_SIZE,
                                        color_mode='grayscale',
                                        class_mode='categorical',
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)

# Define model callback for early stopping
ACCURACY_THRESHOLD = 0.98


class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > ACCURACY_THRESHOLD:
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True


callback = MyCallback()


# Define LeNet-5 model
def lenet():
    filters_no = 60
    filter_size = (5, 5)
    filter_size2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_nodes = 500

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=filters_no, kernel_size=filter_size, activation='relu',
                            input_shape=(32, 32, 1)))

    model.add(layers.Conv2D(filters=filters_no, kernel_size=filter_size, activation='relu'))

    model.add(layers.MaxPooling2D(pool_size=size_of_pool))

    model.add(layers.Conv2D(filters=filters_no // 2, kernel_size=filter_size2, activation='relu'))

    model.add(layers.Conv2D(filters=filters_no // 2, kernel_size=filter_size2, activation='relu'))

    model.add(layers.MaxPooling2D(pool_size=size_of_pool))

    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=no_of_nodes, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(units=classes_no, activation='softmax'))

    return model


# Compile model
model = lenet()
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

start_time = datetime.datetime.now()
print("Training started at: ", start_time)

# Train model
history = model.fit(
      train_data,
      steps_per_epoch=train_data.samples // BATCH_SIZE,  # One pass through entire training dataset
      epochs=1,
      validation_data=val_data,
      validation_steps=val_data.samples // BATCH_SIZE,  # One pass through entire validation dataset
      callbacks=[callback],
      verbose=1)

end_time = datetime.datetime.now()
print("Training ended at: ", end_time)

duration = end_time - start_time
print("Duration: ", duration)


class ModelHistory(object):
    """Save model history"""
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


with open(model_path+'/history', 'wb') as file:
    model_history = ModelHistory(history.history, history.epoch, history.params)
    pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)


# Save model
model.save('trained_model/lenet_acc98')
