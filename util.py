import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

def file_get_contents(filename):
    with open(filename) as f:
        return f.read()


def GetGroundTruthCoordinates(data_path, image_width):
    coordinatesList = []
    amount_of_points = 14
    os.chdir(data_path+"annotations")
    for file in sorted(glob.glob("*.txt")):
        content = file_get_contents(file).split()
        # map string to float
        content = [float(i)*image_width for i in content]
        print(file)
        # filter out the first and last 2 parameters (which are not coordinates)
        coordinates = content[1:]
        coordinates = coordinates[:-2]

        amount_of_points = int(len(coordinates)/2)

        # add to list
        coordinatesList.append(coordinates)

    # Convert to Numpy array
    coordinatesList = np.array(coordinatesList, dtype=None, copy=True, order='K', subok=False, ndmin=0)

    # output some tests to console
    print("")
    print("A test coordinate (the first in the list): ")
    print(coordinatesList[0].reshape((amount_of_points, 2)))
    print("")

    return coordinatesList, amount_of_points


def GetImages(data_path, width, height):
    os.chdir(data_path)
    images = []
    for file in sorted(glob.glob(data_path+"/*.png")):
        print(file)
        img = load_image(file, width, height)
        images.append(img)
    return images

def load_image(infilename, width, height) :
    # Greyscale
    # img = Image.open( infilename ).convert('L')
        
    img = Image.open( infilename )
    img.load()
    
    size = width, height
    img = img.resize(size, Image.ANTIALIAS)
    
    data = np.asarray( img, dtype="int32" )
    return data



def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )


def ExportGroundTruthImageSelection(images, coordinatesList, amount_of_points, output_path):
    first_images = images[:25]

    nrows=5
    ncols=5
    fig, axes = plt.subplots(figsize=(50, 50), nrows=nrows, ncols=ncols)

    counter = 0
    for image, ax in zip(first_images, axes.ravel()):
    
        xy = coordinatesList[counter].reshape((14, 2))
        #   ax.imshow(image, cmap='gray')
        ax.imshow(image)
        ax.axis('off')
        ax.plot(xy[:, 0], xy[:, 1], 'ro')
        counter = counter +1
    plt.savefig(output_path+'ground_truth_preview.png')

def GetData(images, coordinatesList):
    X = np.stack(images).astype(np.float)[:, :, :, :]
    y = np.vstack(coordinatesList)

    print('')
    print('X.shape:')
    print(X.shape, X.dtype)
    print('y.shape:')
    print(y, y.dtype)
    print('')

    output_pipe = make_pipeline(
        MinMaxScaler(feature_range=(-1, 1))
    )

    X_train = X / 255
    y_train = output_pipe.fit_transform(y)

    return X_train, y_train, output_pipe, X

def plot_faces_with_keypoints_and_predictions(model, X_train, X, output_pipe, output_path, model_input='flat', nrows=5, ncols=5):
    """Plots sampled faces with their truth and predictions."""
    selection = np.random.choice(np.arange(X.shape[0]), size=(nrows*ncols), replace=False)
    fig, axes = plt.subplots(figsize=(50, 50), nrows=nrows, ncols=ncols)
    for ind, ax in zip(selection, axes.ravel()):
        # img = X_train[ind, :, :, 0]
        img = X_train[ind, :, :, :]
        if model_input == 'flat':
            predictions = model.predict(img.reshape(1, -1))
        else:
        # predictions = model.predict(img[np.newaxis, :, :, np.newaxis])
            predictions = model.predict(img[np.newaxis, :, :, :])
        
        xy_predictions = output_pipe.inverse_transform(predictions).reshape(14, 2)
        # ax.imshow(img, cmap='gray')
        ax.imshow(img)
        ax.plot(xy_predictions[:, 0], xy_predictions[:, 1], 'bo')
        ax.axis('off')
    plt.savefig(output_path+'results_first25images.png')

def PlotHistory(history, output_path):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(output_path+'history_plot.png')