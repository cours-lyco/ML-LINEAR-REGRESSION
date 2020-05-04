#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

##################################################################################
#                                  MATH PHYSIC CODE: 12-02-2020
##################################################################################

class SimpleRegression(object):
    def __init__(self, height, width, slope_intercetpt_choice=1):
        self.height = height 
        self.width = width
        self.X = []
        self.y = []
        self.choice = slope_intercetpt_choice #1=covariance method, 2=matrix
        self.a, self.b = None, None


        self.initUI()

    def initUI(self):
        self.myCanvas = tk.Canvas(root, bg="black", height=500, width=700)
        

        #self.draw_point(self.width-5, 67)
        self.myCanvas.bind('<Button-1>', self.draw_point)
        #self.draw()
        self.myCanvas.pack()
        

    def draw_point(self, event):
        x = event.x 
        y = event.y 
        self.myCanvas.create_oval(x, y, x + 8, y + 8, fill="white")
        (self.X).append(x)
        (self.y).append(y)
        self.draw()
    
    

    def draw(self):
        self.redraw()
        if(2 == len(self.X)):
            x0, y0, x1, y1 = (self.X)[0], (self.y)[0], (self.X)[1], (self.y)[1]
            
            self.myCanvas.create_line(x0,y0,x1, y1, fill="red")
        else:
            self.prediction()

    def linear_coef(self,choice=1):  #1= covariance methode, 2=matrix_methode
        #a = (cov(x,y)/V(x) = (1/N)*sigma(xi,yi) - xbar * ybar)/( cov(x,y)/V(x) = (1/N)*sigma(xi,xi) - xbar * xbar)
        #b = ybar - a * xbar
        xysum, xxsum, xsum, ysum, M = 0,0,0,0, len(self.X)
        X, y = self.X, self.y

        if len(self.X) < 2:
            return None, None
        a,b = 0,0
        for i in range(M):
            xysum += (X[i]*y[i])
            xxsum += (X[i] ** 2)
            xsum += X[i]
            ysum += y[i]
        if self.choice == 1 :
            xbar, ybar = xsum/ M, ysum/M
            a = ((0.5/M)*xysum - xbar*ybar)/((0.5/M)*xxsum - xbar * xbar)
            b = ybar - a * xbar
            #return a,b 
        if self.choice == 2 :
            n, sumXi , sumXiCarre= len(self.X), xsum, xxsum 
            sumYi , sumXiYi= ysum, xysum 
            matrixLeft = np.array([n, sumXi, sumXi, sumXiCarre
            ]).reshape(2,2)

            matrixRight = np.array([sumYi, sumXiYi]).reshape(2,1)
            resultMatrix = np.matmul(np.linalg.inv(matrixLeft), matrixRight).reshape(1, -1)
            
            resultMatrix = resultMatrix.reshape(1, 2)
            resultMatrix = resultMatrix[0]
            
            a,b = resultMatrix[1], resultMatrix[0]
        return a,b
    
    def prediction(self):
        a, b  = self.linear_coef()

        if a is None and b is None:
            return
        self.a = a 
        self.b = b
        M = len(self.X)

        predict = [(a * xi + b) for xi in self.X]
        x0,y0 = self.X[0], predict[0]
        for i in range(1, M):
                x1, y1 = self.X[i], predict[i]
                self.myCanvas.create_line(x0, y0, x1,y1, fill="red")
                x0, y0 = x1, y1

    def redraw(self):
         self.myCanvas.delete("all")
         for i in range(len(self.X)):
             x, y = (self.X)[i], (self.y)[i]
             regionColor = self.get_regionColor(x, y)
             self.myCanvas.create_oval(x, y, x + 8, y + 8, fill=regionColor)
    
    def get_regionColor(self,x,y):
        if (self.a != None) and (self.b != None):
            fx = (self.a * x ) + self.b - y 
            if fx > 0:
                return 'blue'
            return 'green'
        return 'white'

        


if __name__ == '__main__':
    root = tk.Tk()
    T = tk.Text(root, height=2, width=70)
    T.pack()
    T.insert(tk.END, "         --->> MathsPhysic Code:<<-- ARTIFICIAL INTELLIGENCY\n             [ Linear Regression ] coded in python\n")
    
    sr = SimpleRegression(2000, 1200)
    
    root.mainloop()
   
  
