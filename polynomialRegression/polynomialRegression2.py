#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

##################################################################################
#                                  MATH PHYSIC CODE: 17-02-2020
##################################################################################

class PolynomialRegression(object):
    def __init__(self, height, width):  #Poly degre 2
        self.height = height 
        self.width = width
        self.X = []
        self.y = []
        self.a, self.b , self.c = None, None, None


        self.initUI()

    def initUI(self):
        self.myCanvas = tk.Canvas(root, bg="#000000", height=500, width=700)
        

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
        if(len(self.X) > 3):
            self.prediction()

    def linear_coef(self,choice=1):  #1= covariance methode, 2=matrix_methode
        #a = (cov(x,y)/V(x) = (1/N)*sigma(xi,yi) - xbar * ybar)/( cov(x,y)/V(x) = (1/N)*sigma(xi,xi) - xbar * xbar)
        #b = ybar - a * xbar
        xysum, xxsum, xxxsum , xxxxsum, xsum, n = 0,0,0, 0,0, len(self.X)
        X, y = self.X, self.y
        ysum, xxysum = 0, 0

        if len(self.X) < 4:
            return None, None, None
       
        for i in range(n):
            xysum += (X[i]*y[i])

            xxsum += (X[i] ** 2)
            xxxsum += (X[i] ** 3)
            xxxxsum += (X[i] ** 4)

            xsum += X[i]
            ysum += y[i]
            xxysum += y[i] * (X[i] ** 2)
        
        
        matrixLeft = np.array([n, xsum, xxsum, 
                              xsum, xxsum, xxxsum,
                               xxsum, xxxsum, xxxxsum
            ]).reshape(3,3)

        matrixRight = np.array([ysum, xysum, xxysum]).reshape(3,1)
        resultMatrix = np.matmul(np.linalg.inv(matrixLeft), matrixRight).reshape(1, -1)
            
        resultMatrix = resultMatrix.reshape(1, 3)
        resultMatrix = resultMatrix[0]
       
        a, b, c = resultMatrix[2], resultMatrix[1], resultMatrix[0]
        #self.a, self.b, self.c = a, b, c
        return a,b,c
    
    def prediction(self):
        a, b, c  = self.linear_coef()

        if a is None and b is None:
            return
        self.a = a 
        self.b = b
        self.c = c
        M = len(self.X)

        predict = [((a * (xi ** 2))  + (b * xi) + c) for xi in self.X]
        x0,y0 = self.X[0], predict[0]
        for i in range(1, M):
                x1, y1 = self.X[i], predict[i]
                self.myCanvas.create_line(x0, y0, x1,y1, fill="orange")
                x0, y0 = x1, y1
    
    def redraw(self):
         self.myCanvas.delete("all")
         zipped = zip(self.X, self.y)
         zipped = list(zipped)
         res = sorted(zipped, key=lambda x: x[0])
         self.X, self.y = [x[0] for x in res], [x[1] for x in res]
         for i in range(len(self.X)):
             x, y = (self.X)[i], (self.y)[i]
             #regionColor = self.get_regionColor(x, y)
             self.myCanvas.create_oval(x, y, x + 8, y + 8, fill=self.get_regionColor(x,y), dash=(4,2))
    
    
    def get_regionColor(self,xi, yi):
        if (self.a != None) and (self.b != None) and (self.c != None):
            y = ((self.a * (xi ** 2)) + (self.b * xi) + self.c)
            if y < yi:
                return 'red'
            return 'white'
        return 'white'

        


if __name__ == '__main__':
    root = tk.Tk()
    T = tk.Text(root, height=2, width=70)
    T.pack()
    T.insert(tk.END, "          MathsPhysic Code:ARTIFICIAL INTELLIGENCY\n           Polynomial regression coded in python\n")
    sr = PolynomialRegression(2000, 1200)
    
    root.mainloop()
   
  
