import platform
import shutil
import sys
import time

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog
from gui.dust_detective import Ui_MainWindow
import os
import imutils
from proc.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from proc.preprocessing.preprocess import SimplePreprocessor
from proc.datasets.dataset_loader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import freetype
import tensorflow as tf
import argparse
from PIL import Image, ImageDraw, ImageFont
import cv2

# initialize the class labels
classLabels = []

# parse config
cfg = open('config.csv', 'r').read()
cfg = cfg.split(os.linesep)
for c in cfg:
    if len(c) > 0:
        key, val = c.split(',', maxsplit=1)
        if key == 'modelPath':
            model_ = val
        elif key == 'classIndex':
            continue
        else:
            classLabels.append(val)
    else:
        break
classLabels = np.array(classLabels)


def get_system_type():
    current_system_type = platform.uname()
    if current_system_type.system == "Linux" and current_system_type.machine == "armv7l":
        dev_name = "raspberrypi"
    elif current_system_type.system == "Darwin":
        dev_name = "mac"
    elif current_system_type.system == "Linux" and current_system_type.machine != "armv7l":
        dev_name = "linux"
    elif current_system_type.system == "Windows":
        dev_name = "windows"
    return dev_name


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.predict_thread = Predict()
        self.predict_thread.start()

        self.loadImg_.clicked.connect(self.image_load)
        self.saveImg_.clicked.connect(self.image_save)
        self.clear_.clicked.connect(self.clear_all)

        self.qpix_with_label = None
        self.qpix = None
        self.qpix_with_all = None
        self.qpix_with_proba = None
        self.addLabel_.stateChanged.connect(self.predict_thread.change_qpix)
        self.rank5_acc.stateChanged.connect(self.predict_thread.change_qpix)

        self.probability = np.zeros([5, 2], dtype=np.str)

        self.i = 0
        self.files_len = 0
        self.next_.clicked.connect(self.press_next)
        self.previous_.clicked.connect(self.press_previous)

        self.img_save_list = None
        self.file_list = []

        with tf.device('/cpu:0'):
            # load the pre-trained network
            print("[INFO] loading pre-trained network...")
            self.model = load_model(model_)

    def press_next(self):
        self.i += 1
        if self.i >= self.files_len:
            self.i = self.files_len - 1

        self.predict_thread.load_img(self.i)

    def press_previous(self):
        self.i -= 1
        if self.i < 0:
            self.i = 0

        self.predict_thread.load_img(self.i)

    def clear_all(self):
        self.image_.setDisabled(True)
        self.qpix = None
        self.qpix_with_label = None
        self.i = 0

    def image_load(self, event):
        self.i = 0

        files, ok = QFileDialog.getOpenFileNames(self,
                                                 "Select image(s)",
                                                 "./",
                                                 "Image Files (*.jpg *.png);;All Files (*)")
        if ok:
            files = [f.replace('\\', '/') for f in files]
            self.files_len = len(files)
            self.img_save_list = np.zeros(self.files_len, dtype=object)
            self.file_list = files
            self.predict_thread.predict(files)
            return files
        else:
            return False

    def image_save(self, event):
        dir = QFileDialog.getExistingDirectory(self,
                                                 "Select image(s) save path",
                                                 "./")
        if len(dir) > 0:
            for i, image in enumerate(self.img_save_list):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                path = dir + '/' + self.file_list[i].split('/')[-1].split('.')[0] + '_marked.jpg'
                if os.path.exists(path):
                    os.remove(path)
                cv2.imwrite(path, image)
            return True
        else:
            return False


    def img2qpix(self, image, fit_to):
        # OpencCV numpy array cvt To QImage
        height, width, channel = image.shape
        pile_width = channel * width
        # adapt window
        monitor_width, monitor_height = fit_to.width(), fit_to.height()
        monitor_ratio = monitor_width / monitor_height
        video_ratio = width / height
        # considering calculation precision error issue, don't use "!=" to judge
        if monitor_ratio - video_ratio > 0.1:
            image = imutils.resize(image, height=monitor_height, inter=cv2.INTER_AREA)
            height, width, channel = image.shape
            pile_width = channel * width
            q_image = QImage(image.data, width, height, pile_width, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(q_image).scaled(width, height, Qt.KeepAspectRatio)
        elif monitor_ratio - video_ratio < 0.1:
            image = imutils.resize(image, width=monitor_width, inter=cv2.INTER_AREA)
            height, width, channel = image.shape
            pile_width = channel * width
            q_image = QImage(image.data, width, height, pile_width, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(q_image).scaled(width, height, Qt.KeepAspectRatio)
        return qpix


class Predict(QThread):
    def __init__(self):
        super(Predict, self).__init__()

    def predict(self, files):
        print("[INFO] loading images...")
        self.imagePaths = np.array(files).astype(str)

        # initialize the image preprocessors
        sp = SimplePreprocessor(224, 224)
        iap = ImageToArrayPreprocessor()

        # load the dataset from disk then scale the raw pixel intensities
        # to the range [0, 1]
        sdl = SimpleDatasetLoader(preprocessor=[sp, iap], gray=False)
        data, labels = sdl.load(self.imagePaths, verbose=500)
        data = data.astype(np.float) / 255

        # make predictions on the images
        print("[INFO] prediction...")
        preds = mainWindow.model.predict(data, batch_size=32)
        self.preds_rank1_label = preds.argmax(axis=1)
        self.preds_rank_all_label = np.argsort(preds, axis=1)[:, ::-1]  # from big to small
        self.preds_rank1_proba = preds.max(axis=1)
        self.preds_rank_all_proba = np.sort(preds, axis=1)[:, ::-1]  # from big to small

        for i in range(len(self.imagePaths)):
            self.load_img(i)

        self.load_img(0)

    def load_img(self, i):

        if i < 0:
            i = 0
        elif i >= len(self.imagePaths):
            i = len(self.imagePaths) - 1

        # loop over the sample images
        imagePath = self.imagePaths[i]
        # get rank-5 acc
        rank_label = classLabels[self.preds_rank_all_label[i, :][:5]]
        rank_proba = self.preds_rank_all_proba[i, :][:5]
        mainWindow.probability = np.transpose(np.vstack((rank_label, rank_proba)))

        proba = ""
        proba_list = []
        for l, p in zip(rank_label, rank_proba):
            data = l + ', ' + '{:.3f}%'.format(p * 100)
            proba += data + os.linesep
            proba_list.append(data)

        mainWindow.result_.setText(proba)

        # load the example image, draw the prediction, and display it
        # to our screen
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=640)

        cv2.putText(image, 'AI Image Classification Tech Produced by Carl.Cheung', (10, 60),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            pilimg = Image.fromarray(image)
            draw = ImageDraw.Draw(pilimg)
            ttf_ = 'Arial Unicode.ttf' if get_system_type() == 'mac' else 'msyh.ttc'
            font = ImageFont.truetype(ttf_, 15, encoding="utf-8", layout_engine=ImageFont.LAYOUT_RAQM)
            draw.text((10, 70), "人工智能图像分类技术由张家豪提供", (0, 255, 255), font=font)
            image = np.array(pilimg)
        except:
            pass

        image_with_label = image.copy()
        cv2.putText(image_with_label, "Label: %s" % classLabels[self.preds_rank1_label[i]], (10, 30),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

        image_with_proba = image.copy()
        for j in range(5):
            cv2.putText(image_with_proba, proba_list[j], (10, image.shape[0] - 150 + 25 * j),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)

        image_with_all = image_with_label.copy()
        for j in range(5):
            cv2.putText(image_with_all, proba_list[j], (10, image.shape[0] - 150 + 25 * j),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)

        mainWindow.qpix_with_label = mainWindow.img2qpix(image_with_label, fit_to=mainWindow.image_)
        mainWindow.qpix_with_proba = mainWindow.img2qpix(image_with_proba, fit_to=mainWindow.image_)
        mainWindow.qpix_with_all = mainWindow.img2qpix(image_with_all, fit_to=mainWindow.image_)
        mainWindow.qpix = mainWindow.img2qpix(image, fit_to=mainWindow.image_)

        self.image_label = image_with_label
        self.image_proba = image_with_proba
        self.image_all = image_with_all
        self.image = image

        self.change_qpix()
        self.save_img(i)

        if i < len(self.imagePaths) - 1:
            mainWindow.image_next_.setEnabled(True)
            mainWindow.image_next_.setPixmap(
                mainWindow.img2qpix(cv2.cvtColor(cv2.imread(self.imagePaths[i + 1]), cv2.COLOR_BGR2RGB),
                                    fit_to=mainWindow.image_next_))
        else:
            mainWindow.image_next_.setDisabled(True)

    def change_qpix(self):
        if mainWindow.qpix_with_label is not None and mainWindow.qpix is not None and mainWindow.qpix_with_all is not None:
            mainWindow.image_.setEnabled(True)
            if mainWindow.rank5_acc.isChecked() and mainWindow.addLabel_.isChecked():
                pixmap = mainWindow.qpix_with_all
            elif mainWindow.rank5_acc.isChecked():
                pixmap = mainWindow.qpix_with_proba
            elif mainWindow.addLabel_.isChecked():
                pixmap = mainWindow.qpix_with_label
            else:
                pixmap = mainWindow.qpix

            mainWindow.image_.setPixmap(pixmap)

    def save_img(self, index):
        if mainWindow.rank5_acc.isChecked() and mainWindow.addLabel_.isChecked():
            image = self.image_all
        elif mainWindow.rank5_acc.isChecked():
            image = self.image_proba
        elif mainWindow.addLabel_.isChecked():
            image = self.image_label
        else:
            image = self.image

        mainWindow.img_save_list[index] = image


# setup GUI
app = QApplication(sys.argv)
mainWindow = MainWindow()
mainWindow.show()

sys.exit(app.exec_())
