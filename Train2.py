import os
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

train_path = 'Data/Train'

valid_path = 'Data/Val'

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 20,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory(directory = train_path,
                                                 target_size = (256,256),
                                                 classes = ['Metal','Organic','Paper','Plastic'],
                                                #  batch_size = 10,
                                                 shuffle = True, class_mode = 'categorical')

valid_gen = valid_datagen.flow_from_directory(directory = valid_path,
                                                 target_size = (256,256),
                                                 classes = ['Metal','Organic','Paper','Plastic'],
                                                #  batch_size = 10,
                                                 shuffle = False, class_mode = 'categorical')

print("Train/Validation indicies: ", train_gen.class_indices)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2

model = Sequential()

N = 32
model.add(Conv2D(N,(3,3), activation='relu', padding='same',  input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

while(N!=256):
    N = 2*N
    model.add(Conv2D(N,(3,3), activation = 'relu', padding='same', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu', kernel_regularizer = L1L2(l2=0.001)))
model.add(Dense(128, activation = 'relu', kernel_regularizer = L1L2(l2=0.001)))
model.add(Dense(4, activation = 'softmax'))
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

best_model = ModelCheckpoint('bestmodel.hdf5',monitor='val_loss', save_best_only=True)

history = model.fit(train_gen,validation_data=valid_gen, epochs = 50, callbacks=[best_model, early_stopping])
