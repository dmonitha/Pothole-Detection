import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, GlobalMaxPooling2D, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential

import tensorboard

import json

datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=60, shear_range=20)

PATH = "./dataset/"
train_gen = datagen.flow_from_directory(PATH, target_size=(225,225), class_mode='binary', classes=['normal', 'potholes'])
validation_gen = datagen.flow_from_directory(PATH, target_size=(225,225), class_mode='binary', classes=['normal', 'potholes'], subset='validation')

resnet_model = keras.applications.ResNet50V2(weights='imagenet', input_shape=(225, 225, 3), include_top=False)

for layer in resnet_model.layers:
    layer.trainable=False

model = Sequential([
    Conv2D(64, kernel_size=(2, 2), input_shape=(225, 225, 3), padding='same'),
    Activation('relu'),
     BatchNormalization(),
     MaxPool2D(pool_size=(2,2),  strides=(2,2)),
    
     Conv2D(128, kernel_size=(2, 2), padding='same'),
     Activation('relu'),
     BatchNormalization(),
     MaxPool2D(pool_size=(2,2),  strides=(2,2)),
    
     Conv2D(256, kernel_size=(2, 2), padding='same'),
     Activation('relu'),
     BatchNormalization(),
     MaxPool2D(pool_size=(2,2),  strides=(2,2), padding='same'),
    
    Conv2D(256, kernel_size=(2, 2)),
    Activation('relu'),
     BatchNormalization(),
     MaxPool2D(pool_size=(2,2),  strides=(2,2), padding='same'),
    resnet_model,
    Flatten(),
    Dense(256),
    Dropout(0.5),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid')
    
])

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save Model Sturcture
model_stucture_save_path = "./Pothole_Model_Structure.json"
model_json = model.to_json()
with open(model_stucture_save_path, 'w') as json_file:
    json_file.write(model_json)

    # Callbacks
model_save_path = './Pothole.h5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, mode='max', monitor='accuracy', save_best_only=True, save_weights_only=True, verbose=1)

tensorboard_checkpoint = keras.callbacks.TensorBoard(log_dir="./logs/", write_graph=True, write_images=True, update_freq='batch')
!rm -rf ./logs/*

 # %reload_ext tensorboard
# %load_ext tensorboard
# %tensorboard --logdir=./logs/
history = model.fit(train_gen,
                    validation_data=validation_gen,
                    epochs=20,
                    steps_per_epoch=int(580/32),
                    validation_steps=int(101/32),
                    callbacks=[checkpoint, tensorboard_checkpoint],
                    verbose=1
                   )
 ?keras
images = next(train_gen)
len(images)
