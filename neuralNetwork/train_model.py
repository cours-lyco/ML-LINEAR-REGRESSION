#!/usr/bin/env python3

import os
#from keras.datasets import mnist
import tensorflow as tf
#import numpy as np
from PIL import Image
import os
import sys
import glob

from tensorflow import keras
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer, UpSampling2D, Conv2D, MaxPooling2D

from tensorflow.keras import backend as K

from keras.utils import to_categorical
import numpy as np
from keras.models import load_model

import plaidml.keras
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#Python --version: Python 3.7 :: Anaconda, Inc.
'-------------------------------------------------------'
'               EXTERNAL PATHS                          '
'-------------------------------------------------------'
'''path of external device '''
rootDir = "/Volumes/Disk E/"
datasDir = rootDir + "/crohme2016_inkml_datasets/CROHME2016_data/"

imagesDir = rootDir + "/crohme2016_inkml_images/"
'''now get train valid test dir'''

trainDir = datasDir + 'Task-2-Symbols/task2-trainSymb2014/'
trainDirDatasets = trainDir + 'trainingSymbols/'
trainDir_ground_truth = trainDir + 'trainingSymbols/iso_GT.txt'
trainDatasImagesDir = "/Volumes/Disk E/crohme2016_inkml_images/train_images"

testDir = datasDir + 'Task-2-Symbols/task2-testSymbols2014/'
testDir_ground_truth = testDir + "testSymbols_2016_iso_GT.txt"
testDirDatasets = testDir + "testSymbols/"
testDatasImagesDir = "/Volumes/Disk E/crohme2016_inkml_images/test_images"

valDir = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDirDatasets = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDir_ground_truth = valDirDatasets + 'iso_GT.txt'
valDatasImagesDir = "/Volumes/Disk E/crohme2016_inkml_images/val_images"

models_save = "/Volumes/Disk E/models_save/"
canvas_stroke = "/Volumes/Disk E/canvas_stroke/"
'--------------------------------------------------------------'
#from creat_imkml_images import *
from ground_truth import *


class CNN(GT):

    differents_characters = 101
    image_len = 28
    X_train_len = 85802
    X_test_len = 18435
    X_val_len = 12504

    def __init__(self):
        super().__init__()
        self.X_train, self.X_test, self.X_val = [], [], []
        self.model = None
        #self.y_train, self.y_test, self.y_val = [], [], []

        #self.y_train_character_ordered = []
        #self.y_test_character_ordered = []
        #self.y_val_character_ordered = []

        #self.gt = [ {sortedDictTrain}  , {sortedDictTest}, {sortedDictval} ]
        #sortedDictTrain = ')' : [chemin de ttes les images de parenthèses], '9': [chemin de ttes les images de 9],  ...
        #self.gt = {}

        '''second load images and prevent junk images  to load '''
        
        #########  LOAD ALL IMAGES MATRIX 
        self.load_all_images_Matrix()

        self.X_train, self.X_test, self.X_val = np.array(
                self.X_train), np.array(self.X_test), np.array(self.X_val)
        #self.y_train, self.y_test, self.y_val = np.array(
                #self.y_train), np.array(self.y_test), np.array(self.y_val)

        self.X_train, self.X_test,  self.X_val = (self.X_train - np.mean(self.X_train)) / np.std(self.X_train), (self.X_test - np.mean(self.X_test)) / np.std(
                self.X_test), (self.X_val - np.mean(self.X_val)) / np.std(self.X_val)
        
        ######### TRAIN MODEL
        self.train_and_save_model()
    
    def get_matrix_from_image(self, infilename):
        img = Image.open(infilename)
        img.load()
        imageMatrix = np.asarray(img, dtype="float32")/255.0
        return imageMatrix
   
    #------------------------------------------------------------------------------------------------------#
    # result: X_train = [imageMatrix_iso0.png, imageMatrix_iso0.png, ........., imageMatrix_iso85802.png ]
    #         X_test = [imageMatrix_iso0.png, imageMatrix_iso0.png, ........., imageMatrix_iso18402.png  ]
    #         X_val = [imageMatrix_iso0.png, imageMatrix_iso0.png, ........., imageMatrix_iso18402.png ]
    #--------------------------------------------------------------------------------------------------------#

    def load_all_images_Matrix(self):
        directories = [trainDatasImagesDir,
                       testDatasImagesDir, valDatasImagesDir]
        if not self.gtruth:
            print("Error no ground truth to prevent to load junk image")
            return
        for index, data_dir_abs_path in enumerate(directories):
            print("-->>-->>-->> loding images  matrix <<--<<--<<--\n")
            if os.path.isdir(data_dir_abs_path):
                if index == 0:
                    slice_start, slice_stop = 3, -4
                    ground_truth = self.gtruth['train_gt']
                if index == 1:
                    slice_start, slice_stop = 3, -4
                    ground_truth = self.gtruth['test_gt']
                if index == 2:
                    slice_start, slice_stop = 4, -4
                    ground_truth = self.gtruth['val_gt']

                print(".... start loading  in folder: {}".format(data_dir_abs_path))
                #get array of files
                images_files_array = glob.glob(data_dir_abs_path + os.sep + "*.png")

                #sort files array
                images_files_array = sorted(images_files_array, key=lambda name: int(
                    (os.path.basename(name))[slice_start: slice_stop]))

                #inkml_files = inkml_files.sort(key=self.sortGlobFilesArray)
                for j, image_file_abs_path in enumerate(images_files_array):

                    file_name_index = image_file_abs_path.split('/')[-1]
                    file_name_index = file_name_index[int(
                        slice_start): int(slice_stop)]
                    #print(".............fle check: {} .............".format(file_name_index))
                    file_is_junk = True
                    
                    for key in ground_truth.keys():
                        if int(file_name_index) in ground_truth[key]: 
                            file_is_junk = False
                            if False == file_is_junk:
                                imageMatrix = self.get_matrix_from_image(
                                image_file_abs_path).reshape(28, 28, 1)

                                if index == 0:
                                    self.X_train.append(imageMatrix)
                                if index == 1:
                                    self.X_test.append(imageMatrix)
                                if index == 2:
                                    self.X_val.append(imageMatrix)
                    #else:
                        #print('...............Found JUNK FILE .....................\n')
                '--------------------------------------------------------------------'

                '--------------------------------------------------------------------'
                if index == 0:
                    print(".... ALL self.X_train IMAGES LOADED:  ....{} matrix".format(
                        len(self.X_train)))
                if index == 1:
                    print(".... ALL self.X_test IMAGES LOADED:  ....{} matrix".format(
                        len(self.X_test)))
                if index == 2:
                    print(".... ALL self.X_val IMAGES LOADED:  ....{} matrix".format(
                        len(self.X_val)))
            else:
                print("Error: External Devices not found\n")

    def  train_and_save_model(self):
        self.create_model()
        self.compile_model()
        self.train_model()
        self.save_model()

   
    #---------------------------------------------------------------------------------------------------#

    def create_model(self):
        print("-------->> Start creating model ----------\n")
        model = Sequential()
        model.add(InputLayer(input_shape=self.X_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2,
                         kernel_initializer='truncated_normal'))
        model.add(MaxPooling2D((3, 3), strides=1, padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2,
                         kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1,
                         kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2,
                         kernel_initializer='truncated_normal'))
        model.add(MaxPooling2D((3, 3), strides=1, padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1,
                         kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1,
                         kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1,
                         kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1,
                         kernel_initializer='truncated_normal'))
        model.add(Flatten())
        model.add(Dense(101, activation='softmax'))
        #model.add(UpSampling2D((2, 2)))
        self.model = model
        print("-------->> End creating model ----------\n")

    def compile_model(self):
        print("-------->> Start compiling model ----------\n")
        #compile model using accuracy to measure model performance
        # Compile the model
        #model.compile(
        #loss=keras.losses.mean_squared_error, optimizer='sgd')
        optimizer = tf.keras.optimizers.RMSprop(lr=0.0005, decay=1e-5)
        (self.model).compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        print("-------->> End model compilation  ----------\n")
       
    
    def train_model(self):
        print("-------->> Start traing model ----------\n")
        #fit the model
        #model.fit(self.X_train, self.y_train, batch_size=500, epochs=10)
        # Evaluate the model on test set
        #score = model.evaluate(self.X_test, self.y_test, verbose=0)
        # Print test accuracy
        #print('\nTest accuracy: {}'.format(score[1]*100))
       
       
        (self.model).fit(self.X_train, self.y_categorical['train_y'], batch_size=500,
                  validation_data=(self.X_val, self.y_categorical['val_y']), epochs=50)
        print("-------->> End training model ----------\n")

    def save_model(self):
        # save model and architecture to single file
        (self.model).save(models_save + "model_2.h5")
        print("Saved model to disk: [name: {}]".format(
            models_save + "model_2.h5\n"))
       


if __name__ == '__main__':
    cnn = CNN()

    print("X_train_shape:\n", cnn.X_train.shape)
    print("X_test_shape:\n", cnn.X_test.shape)
    print("X_val_shape:\n" ,cnn.X_val.shape)

    print("Y_train_shape:\n", cnn.y_categorical['train_y'].shape)
    print("Y_test_shape:\n", cnn.y_categorical['test_y'].shape)
    print("Y_val_shape:\n", cnn.y_categorical['val_y'].shape)

    #print("X_train_keys:\n", cnn.y_train_character_ordered)
    #print("X_test_keys:\n", cnn.y_test_character_ordered)
    #print("X_val_keys:\n", cnn.y_val_character_ordered)'''
    
    #print("X_train_0\n", cnn.X_train[0])
    #print("y_train_0\n", cnn.y_train[0])
    
    #print("X_test_0\n", cnn.X_test[0])
    #print("y_tesy_0\n", cnn.y_test[0])
    
    #print("X_val_0\n", cnn.X_val[0])
    #print("y_val_0\n", cnn.y_val[0])

    #model = cnn.create_model()
    #model = cnn.compile_model()
    #model = cnn.train_model()
    
    

