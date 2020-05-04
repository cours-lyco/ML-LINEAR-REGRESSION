#!/usr/bin/env python3
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import linalg
import numpy as np
import tkinter as tk

##################################################################################
#                                  MATH PHYSIC CODE: 13-02-2020
##################################################################################

class PolynomialRegression(object):
    def __init__(self, height, width, polyDegre=2):
        self.height = height 
        self.width = width
        self.polyDegre = polyDegre

        self.X = []
        self.y = []
       


        self.initUI()

    def initUI(self):
        self.myCanvas = tk.Canvas(root, bg="black", height=500, width=700)
        self.myCanvas.bind('<Button-1>', self.draw_point)
        self.myCanvas.pack()
        

    def draw_point(self, event):
        x = event.x 
        y = event.y 
        self.myCanvas.create_oval(x, y, x + 8, y + 8, fill="white")
        (self.X).append(x)
        (self.y).append(y)
        self.draw()
    
    def data_normalize(self):
        # xi = (xi - xmean)/(xmax - xmin)
        X , y = self.X , self.y
        X = np.divide( np.array(X)  , self.width  )
        y = np.divide( np.array(y)  ,self.y  )
        return X.tolist(), y.tolist()
    
    def calculate_matrix_granX(self): 
        #https://www.ritchieng.com/multi-variable-linear-regression/# 
        #https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/polynomial_regression.py#
        M = len(self.X)

        if(M > self.polyDegre):
            #self.X, self.y = self.data_normalize()
            matrixGrandX = np.ones(M).reshape(-1,1)
            for i in range(1, self.polyDegre + 1):
                matrixGrandX = np.concatenate( (matrixGrandX, np.power(self.X, i).reshape(-1,1))  , axis=1)
            return matrixGrandX 
        return None

    def calculate_theta(self):
        #https://www.ritchieng.com/multi-variable-linear-regression/#
        #https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/polynomial_regression.py#
        #theta = inverse( (matrixX_tranpose* matrixX) ) * matrixX_transpose * y
        M = len(self.X)
        
        d = {}
        d['x' + str(0)] = np.ones([1, M])[0]
        for i in np.arange(1, self.polyDegre + 1):
            d['x' + str(i)] = self.X ** (i)

        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        X = np.column_stack(d.values())

        theta = np.matmul(np.matmul(linalg.pinv(
                np.matmul(np.transpose(X), X)), np.transpose(X)), self.y)
        return theta

    def draw(self):
        
        X = self.X
        canvas = self.myCanvas

        theta = self.calculate_theta()
        if theta is None:
            return 
        predict = self.predict(theta)
        return
        if(predict is None):
            return 
        if(2 < len(predict)):
            x0,y0 = X[0], predict[0]
            for i in range(1, len(predict)):
                if x0 >= 0 and y0 >= 0:
                    x1,y1 = X[i], predict[i]
                    if x1 >= 0 and y1 >= 0:
                        canvas.create_line(x0, y0, x1, y1, fill="red")
                        x0,y0 = x1, y1


    def predict(self, theta):
        theta = theta.reshape(1, -1)
        if theta is None:
            return 
        #print(theta)
        predict = []
        #print(theta)
        #print("-------------------------")
        #print(self.X)
        #print(":::::::::::::::::::::::::::::::::::::::::")
        
        for i in range(len(self.X)):
            y = 0
            X = [(self.X)[i]**k for k in range(1 + self.polyDegre)]
            #X.insert(0, 1)
            #print(X)
            #j = 0
            for index in range(len(theta)):
                y += theta[index] * X[index]
            predict.append(y)
            print("predic: ", predict)
        return predict
    
    

if __name__ == '__main__':
    root = tk.Tk()
    sr = PolynomialRegression(2000, 1200)
    
    root.mainloop()
   
  
