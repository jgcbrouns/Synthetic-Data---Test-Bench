import pandas as pd
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from util import *
from model import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense
from keras import optimizers
from keras.layers import Dropout, Flatten
 
from keras.callbacks import CSVLogger

# stuff to use GPU
from tensorflow.python.client import device_lib
from keras import backend as K

############ Parameters ##############
working_dir = "Z:/experiments/"
data_path = "Z:/experiments/data/dataset2/"
results_output_path = "Z:/experiments/results/"

image_width = 400
image_height = 400

epochs = 3
######################################


########### Main Program #############
if __name__ == "__main__":
    print(device_lib.list_local_devices())
    K.tensorflow_backend._get_available_gpus()

    print("Reading coordinates...")
    coordinatesList, amount_of_points = GetGroundTruthCoordinates(data_path, image_width)

    print("Reading images...")
    imagesList = GetImages(data_path, image_width, image_height)

    print("Exporting selection-preview of ground-truth data on images...")
    ExportGroundTruthImageSelection(imagesList, coordinatesList, amount_of_points, results_output_path)

    X_train, y_train, output_pipe, X = GetData(imagesList, coordinatesList)

    model = CNN()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    csv_logger = CSVLogger(results_output_path+'log.csv', append=True, separator=';')

    history = model.fit(X_train, y_train, 
                    validation_split=0.2, shuffle=True, 
                    epochs=epochs, batch_size=20, callbacks=[csv_logger])

    PlotHistory(history, results_output_path)
    plot_faces_with_keypoints_and_predictions(model, X_train, X, output_pipe, results_output_path, model_input='2d')