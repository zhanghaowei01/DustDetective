import os

import imutils

from proc.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from proc.preprocessing.preprocess import SimplePreprocessor
from proc.datasets.dataset_loader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import tensorflow as tf
import argparse
import cv2

__classes__ = 2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained dataset")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ['AA Glue', 'Alumina', 'Black Mask', 'Fiber', 'Glass Fiber', 'IRCF Incoming NG',
               'IRCF Incoming NG(Coating)', 'IRCF Incoming NG(Dig)', 'IRCF Scrap', 'Lens Incoming NG', 'Lens scratch',
               'Not found', 'Organic', 'PP', 'SUS(Cr&Ni,GlueShape)', 'Skin', 'VCM Base'
               ]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = []
pl = os.walk(args["dataset"])
for root, dirs, files in pl:
    for file in files:
        if '.jpg' in file or '.png' in file:
            imagePaths.append(root + "/" + file)

imagePaths = np.array(imagePaths)
idxs = np.random.randint(0, len(imagePaths), size=(20,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessor=[sp, iap], gray=False)
data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype(np.float) / 255

with tf.device('/cpu:0'):
    # load the pre-trained network
    print("[INFO] loading pre-trained network...")
    model = load_model(args["model"])

    # make predictions on the images
    print("[INFO] prediction...")
    preds = model.predict(data, batch_size=32).argmax(axis=1)

    # loop over the sample images
    for (i, imagePath) in enumerate(imagePaths):
        # load the example image, draw the prediction, and display it
        # to our screen
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=640)
        cv2.putText(image, "Label: %s" % classLabels[preds[i]], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
