import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
import serial

IOT = serial.Serial('COM', 9600, timeout=.1)

def predictions(frame):
    model = load_model("bestmodel.hdf5")
    img = frame
    # img = load_img("Image.jpg", target_size=(256,256))
    img_ = img_to_array(img)
    img_ = np.expand_dims(img, axis=0)
    result = model.predict(img_)
    # plt.imshow(img)
    Result = 0
    print(result)
    Result+=np.argmax(result)
    print(Result)
    if(Result==1):
        model2 = load_model("bestmodel2.hdf5")
        result = model2.predict(img_)
        print(result)
        Result+=np.argmax(result)
        print(Result)
        if(Result==1):
            IOT.write('Metal')
            # plt.title('Metal')
        elif(Result==2):
            IOT.write('Others')
            # plt.title('Others')
        elif(Result==3):
            IOT.write('Paper')
            # plt.title('Paper')
        elif(Result==4):
            IOT.write('Plastic')
            # plt.title('Plastic')
    else:
        IOT.write('Organic')
        # plt.title('Organic')
    # plt.show()
