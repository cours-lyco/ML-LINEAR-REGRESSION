#!/usr/bin/env python3

###################
#utils:
#http://www.ianschumacher.ca/gesture_recognition/sourcecode.html
###################
import os
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import Image, ImageFilter, ImageDraw
import io
import math
import cv2

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

#from train_model import *
models_save = "/Volumes/Disk E/models_save/"
canvas_stroke = "/Volumes/Disk E/canvas_stroke/"

from  train_model import *

class CANVAS_STROKE(CNN):
    def __init__(self, width, height):
        super(CANVAS_STROKE, self).__init__()
        self.height = height
        self.width = width
        self.xstroke = []
        self.ystroke = []
        self.strokes = []

        self.lineOldx, self.lineOldy = -1, -1
        self.initUI()


    def initUI(self):
        self.myCanvas = tk.Canvas(root, bg="white", height=self.height, width=self.width)
        self.myCanvas.bind('<B1-Motion>', self.draw_shape)
        self.myCanvas.bind('<ButtonRelease-1>', self.getStroke)
        self.myCanvas.pack()

    def draw_shape(self, event):
        x = event.x
        y = event.y
        #self.myCanvas.create_oval(x, y, x + 8, y + 8, fill="white")
        self.draw_line(event)
        (self.xstroke).append(x)
        (self.ystroke).append(y)
        (self.strokes).append({
            'x':x,
            'y':y
        })
        #self.draw()
    
    'utils'
    def draw_line(self, event):
        x, y = event.x, event.y
        if self.lineOldx != -1 and self.lineOldy != -1:
            self.myCanvas.create_line(self.lineOldx, self.lineOldy, x, y, fill="black")
        self.lineOldx, self.lineOldy = x, y

    def getStroke(self, event):
        self.lineOldx, self.lineOldy = -1, -1
        minX,maxX,minY,maxY = min(self.xstroke), max(self.xstroke), min(self.ystroke), max(self.ystroke)
        
        #x = np.array((28, 28), dtype="float32")
        #self.xstroke = np.array([x - minX for x in self.xstroke], dtype="float32")/255.0
        #self.ystroke=np.array([y - minY for y in self.ystroke], dtype = "float32")/255.0
        imageMatrix = self.convertStroke_toPng_matrix()
        
        #imageMatrix = self.get_matrix_from_image(
        #    "/tmp/my_stroke.png")
        #print(imageMatrix)
        #imageMatrix = imageMatrix[:,:,0]
        imageMatrix = np.array(imageMatrix).reshape(28,28,1)
        imageMatrix = np.expand_dims(imageMatrix, axis=0)
        print("img shame = {},  {}".format(imageMatrix.shape, type(imageMatrix)))
        self.drawBoxContainer(minX, minY, maxX, maxY)
        self.make_prediction(imageMatrix)

    def drawBoxContainer(self, minx, miny, maxx, maxy):
        self.myCanvas.create_rectangle(minx, maxy, maxx, miny, outline="#fb0") #fill="green")
        self.xstroke, self.ystroke, self.strokes = [], [], []

    def dist(self, p1 , p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    #Binary Search that gives us fractional index
    def binarySearch(self,arr, d) :
        startIndex = 0
        stopIndex = len(arr) - 1
        middle =math.floor((stopIndex + startIndex) * 0.5)
        while (arr[middle] != d and startIndex < stopIndex):
            if (d < arr[middle]) :
                stopIndex = middle - 1
            elif (d > arr[middle]) :
                startIndex = middle + 1
            
            middle = math.floor((stopIndex + startIndex) * 0.5)
        return (d - arr[middle]) / (arr[middle + 1] - arr[middle]) + middle
    

    #Rescale the gesture to fixed number of points
    def rescaleByDistance(self):
        distArr = []
        tot = 0
        cur = (self.strokes)[0]
        distArr.append(0)
        for i in range(1, len(self.strokes)):
            tot += self.dist((self.strokes)[i], (self.strokes)[i - 1])
            distArr.append(tot)
    
        r  = []
        r.append(self.strokes[0])
        old = (self.strokes)[0]
        for i in range(1, len(self.strokes))-2:
            # TODO
            # There is an error here somewhere, but haven't found it yet so do try/catch for now
            try :
                d = (i / (length - 1)) * tot
                index = self.binarySearch(distArr, d)
                start = math.floor(index)
                end = start + 1
            
                fraction = index - start
                x = ((self.strokes)[end].x - (self.strokes)[start].x) * fraction + (self.strokes)[start].x
                y = ((self.strokes)[end].y - (self.strokes)[start].y) * fraction + (self.strokes)[start].y
                r.append({'x':x, 'y':y})
                old = p
            except:
                r.append(old)
        r.append((self.strokes)[len(self.strokes) - 1])
        return r

    def draw_rescale(self):
        self.myCanvas.delete("all")
        zipped = zip(self.xstroke, self.ystroke)
        zipped = list(zipped)
        res = sorted(zipped, key=lambda x: x[0])
        self.xstroke, self.ystroke = [x[0] for x in res], [x[1] for x in res]
        for i in range(len(self.xstroke)):
            x, y = (self.xstroke)[i], (self.ystroke)[i]
            self.myCanvas.create_oval(
                x, y, x + 8, y + 8, fill='red', dash=(4, 2))

    def convertStroke_toPng_matrix(self):
        ps = self.myCanvas.postscript(colormode='color')
        stroke_img = Image.open(io.BytesIO(ps.encode('utf-8')))
        
        stroke_img.save('/tmp/my_stroke.png')
        stroke_img.save(canvas_stroke + "/my_stroke.png")
        ####
        im = Image.open(canvas_stroke + "/my_stroke.png").convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        # creates white canvas of 28x28 pixels
        newImage = Image.new('L', (28, 28), (255))
        if width > height:  # check which dimension is bigger
            #Width is bigger. Width becomes 20 pixels.
            # resize height according to ratio width
            nheight = int(round((30/width*height), 0))
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
            # resize and sharpen
            img = im.resize((30, nheight), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
            # caculate horizontal pozition
            wtop = int(round(((32 - nheight)/2), 0))
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            #Height is bigger. Heigth becomes 20 pixels.
            # resize width according to ratio height
            nwidth = int(round((30/height*width), 0))
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
            # resize and sharpen
            img = im.resize((nwidth, 30), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
            wleft = int(round(((32 - nwidth)/2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
            #newImage.save("sample.png")
        tv = list(newImage.getdata())  # get pixel values
        #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255-x)/255 for x in tv]
        return tva
        ####
    
    def get_matrix_from_image(self, infilename):
        img = Image.open(infilename)
        img.load()
        imageMatrix = np.asarray(img, dtype="float32")/255.0
        return imageMatrix
    
    def make_prediction(self, imagematrix):
        X = imagematrix
        retrieve_model = tf.keras.models.load_model(models_save + "model.h5")
        retrieve_model.summary()
        #arr = retrieve_model.predict(X)
        arr = retrieve_model.predict(
            X, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
            workers=1, use_multiprocessing=True
        )
        #print("arr: {}\n".format(arr))
        indexMax = np.argmax(arr)
        keys = list((self.gt['train_gt']).keys())
        #print("keys: {}".format(keys))
        print(" indexMax: {}".format(indexMax))
        print("retrive model prediction: {}".format(keys[indexMax]))

    ###############################################################################
    def load_saved_model(self, model):
        (self.model) = load_model(models_save + "model.h5")
        # summarize model.
        (self.model).summary()

    def evaluate_model(self):
        score = (self.model).evaluate(
            self.X_val, self.y_categorical['val_y'], verbose=0)
        print("%s: %.2f%%" % ((self.model).metrics_names[1], score[1]*100))

    def model_predict(self, X):
        #predict first 4 images in the test set
        #np.argmax(result_digit_1 ,axis = 1)
        (self.model).predict(X[:4])

    def whichCharacter(testCharacterIndex=0, datasets='train'):
        gt, X = [], []
        if datasets == 'train':
            gt = self.gtruth['train_gt']
            X = self.X_train
        if datasets == 'test':
            gt = sel.gtruth['test_gt']
            X = self.X_test
        if datasets == 'val':
            gt = self.gtruth['val_gt']
            X = self.X_val
        arr = model.predict(X[testCharacterIndex:1+testCharacterIndex])
        indexMax = np.argmax(arr)
        keys = gt.keys()
        keys = list(keys)
        return keys[indexMax]

if __name__ == "__main__":
    root = tk.Tk()
    strokes = CANVAS_STROKE(600, 400)

    root.mainloop()
