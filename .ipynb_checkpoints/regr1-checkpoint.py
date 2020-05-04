{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from random import seed\n",
    "from random import random\n",
    "import numpy as np\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = [0,   30,  10,   15,  5, 25, 35,  40]\n",
    "y = [4,   1,    2,    2,  3,  1,  0,   1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(X, y, 'go')\n",
    "plt.xlabel(\"abscisses\")\n",
    "plt.ylabel(\"ordonnee\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a=  -0.08\n",
      "b=  3.35\n"
     ]
    }
   ],
   "source": [
    "xsum = 0\n",
    "ysum = 0\n",
    "xysum = 0\n",
    "xxsum = 0\n",
    "\n",
    "N = len(X)\n",
    "for i in range(N):\n",
    "    xsum += X[i]\n",
    "    ysum += y[i]\n",
    "    xysum += (X[i] * y[i])\n",
    "    xxsum += (X[i] * X[i])\n",
    "xbar = xsum/N\n",
    "ybar = ysum/N\n",
    "a = ((xysum/N) - xbar * ybar) / ((xxsum/N) - xbar * xbar)\n",
    "b = ybar - a * xbar\n",
    "\n",
    "ca = a\n",
    "cb = b\n",
    "print(\"a= \", a)\n",
    "print(\"b= \", b)\n",
    "predict = [a*xi+b for xi in X]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x11edfeb38>]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(X, y, 'go')\n",
    "plt.xlabel(\"abscisses\")\n",
    "plt.ylabel(\"ordonnee\")\n",
    "plt.plot(X, predict, color=\"red\", lineWidth=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "learning_rate = 0.1\n",
    "a0,b0 = random(), random()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "def partial_derivate(X, y):\n",
    "    dfa = 0\n",
    "    dfb = 0\n",
    "    for i in range (N):\n",
    "        dfa += ((a*X[i] + b) - y[i])*X[i]\n",
    "        dfb += ((a*X[i] + b )- y[i])\n",
    "       \n",
    "    return dfa, dfb\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def back_propagation(X, y, a, b):\n",
    "    error = 5;\n",
    "    epoch = 0\n",
    "    while( error > 0.001) && (epoch <= 50 )):\n",
    "        for i in range (N):\n",
    "            newa , newb = partial_derivate(X, y)\n",
    "            a = (a - (learning_rate * newa))\n",
    "            b = (b - (learning_rate * newb))\n",
    "        error = ((a*X[i] + b - y[i]) **2) /2 \n",
    "        print(\"Error is : \",error)\n",
    "        epoch ++\n",
    "    return a, b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a, b = back_propagation(X,y, ca, cb)\n",
    "predict = [(a*X[i] + b) for  i in range(len(X))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.xlabel(\"abscisses\")\n",
    "plt.ylabel(\"ordonnee\")\n",
    "plt.plot(X, predict, color=\"red\", lineWidth=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
