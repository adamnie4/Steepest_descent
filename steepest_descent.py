#!/usr/bin/env python
# coding: utf-8

# In[95]:


#############Algorytm Steepest_Descent

from datetime import datetime
import numpy as np 
from sympy import *
import matplotlib.pyplot as plt


def f(x):
    
    return 0.125*x[0]**2 + x[1]**2   

#Gradient Funkcji w punkcie 
def gradient(x_0):
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    
    f = 0.125*x0**2+x1**2
    f_x0 = f.diff(x0)
    f_x1 = f.diff(x1)
    w1 = float(f_x0.evalf(2, subs={x0: x_0[0]}))
    w2 =float(f_x1.evalf(2, subs={x1: x_0[1]}))
    
    grad = (w1,w2)
    return grad

def steepest_descent(f,x_0,a=5,grid_size = 100,K=100):
    datetime.now()
    
    n = len(x_0) 
    
    x_opt = x_0
    f_opt = f(x_0)
    x_hist = np.zeros((K,n))
    f_hist = []
    #Tworze sobie liste historie ktora na poczatku ma same zera bo nic nie ma

    t_eval = 0 
    #Dodaje poczatkowe punkty  do historii
    x_hist[0] = [x_0[0],x_0[1]]
    #dodaje poczatkowa wartosc funkcji w punkcie 
    f_hist.append(f(x_0))
   
    #wyliczenie x_1 do petli 
    for k in range(1,K):
        #Punkt x_1 trzeba wyliczyć z wzoru 
        pusta = []
        gradientt = gradient(x_0)
        
        punkt_x1_1 = x_0[0] - a * gradientt[0]
        punkt_x1_2 = x_0[1] - a * gradientt[1]
        pusta.append(punkt_x1_1)
        pusta.append(punkt_x1_2)
        
        x_1 = lines_search(f,x_0,pusta,grid_size = 100 )
    ###########To sa listy lub tuple i nie dziala 
        if f(x_1) < f_opt:
            x_opt = x_1
            f_opt = f(x_1)
        
        x_hist[k] = [x_1[0],x_1[1]]
        f_hist.append(f(x_1))
        
        x_0 = x_1

def lines_search(f,x_0,x_1,grid_size):
    x_b = [x_0[0],x_0[1]]
    
    for i in range(0,grid_size):
        lista_t = []

        t = (i+1) / grid_size
        
        ###tutaj trzeba bedzie dac petle zeby x_t bylo zmienna 2 elementowa
        for z in range(2):
            x_t = t * x_1[z] + (1-t)*x_0[z]
            lista_t.append(x_t)
        
        if f(lista_t) < f(x_b):
            x_b.clear()
            x_b.append(lista_t[0])
            x_b.append(lista_t[1]) 
        else:
            break
    return x_b
steepest_descent(f,[3,3],0.5,100,100)


# In[76]:


import matplotlib.pyplot as plt

for i in range(len(x_hist)):
    print(i)

#plt.scatter(x, y)
#plt.show()


# In[73]:




