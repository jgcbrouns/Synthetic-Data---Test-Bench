from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense
from keras import optimizers
from keras.layers import Dropout, Flatten

def CNN():
    model = Sequential()

    # input layer
    model.add(BatchNormalization(input_shape=(400, 400, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(24, (5, 5), kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    # layer 2
    model.add(Conv2D(36, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    # layer 3
    model.add(Conv2D(48, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # layer 4
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    # layer 5
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
   
    # layer 6
    model.add(Dense(500, activation="relu"))
    
    # layer 7
    model.add(Dense(90, activation="relu"))
   
    # layer 8
    model.add(Dense(28))

    return model