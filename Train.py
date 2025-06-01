import os
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

train_path = "Data/train"
val_path = "Data/val"
data_gen = ImageDataGenerator(rescale=1./255)

train_gen = data_gen.flow_from_directory(train_path, target_size=(256,256), batch_size=64)
test_gen = data_gen.flow_from_directory(val_path, target_size=(256,256), batch_size=64)

print(train_gen.class_indices)

# import matplotlib.pyplot as plt

# img, _ = next(train_gen)
# plt.figure(figsize=(15,13))
# for i in range(36):
#     ax = plt.subplot(6, 6, i+1)
#     plt.imshow(img[i])
#     if _[i][1] == 0:
#         plt.title("Organic")
#     else:
#         plt.title("Recyclable")
#     plt.axis("off")
# plt.show()
# del img
# del _

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L1L2

model = Sequential()

N = 32
model.add(Conv2D(N,(3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

while(N!=256):
    N = 2*N
    model.add(Conv2D(N,(3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu', kernel_regularizer = L1L2(l2=0.001)))
model.add(Dense(128, activation = 'relu', kernel_regularizer = L1L2(l2=0.001)))
model.add(Dense(4, activation = 'softmax'))
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

best_model = ModelCheckpoint('bestmodel.hdf5',monitor='val_loss', save_best_only=True)

history = model.fit(train_gen,validation_data=test_gen, epochs = 50, callbacks=[best_model, early_stopping])