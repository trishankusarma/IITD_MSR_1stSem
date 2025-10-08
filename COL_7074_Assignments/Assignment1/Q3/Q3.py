#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from model import ModelClass 


# In[2]:


dfX = pd.read_csv('./logisticX.csv', header = None)
dfY = pd.read_csv('./logisticY.csv', header = None)

print("Shape of dfX : ", dfX.shape)
print("Shape of dfY : ", dfY.shape)


# In[3]:


model = ModelClass()
X, Y, theta = model.initializeXYAndParams(dfX, dfY)


# In[4]:


# Question 1
#  maxEpocs=100000, tolerance=1e-9, alpha=0.001
optimalTheta = model.optimizeLikelihoodFunction(theta)
# So the coefficients that I got are :
# Epoch 17313, Loss: 22.834145, Theta: [[ 0.40125296  2.58854703 -2.72558777]]


# In[5]:


# Question 2
# Plot the training data (your axes should be x1 and x2 , corresponding to the two coordinates of
# the inputs, and you should use a different symbol for each point plotted to indicate whether that example
# had label 1 or 0). Also plot on the same figure the decision boundary fit by logistic regression. (i.e.,
# this should be a straight line showing the boundary separating the region where h(x) > 0.5 from where
# h(x) ≤ 0.5.)
model.plotDecisionBoundary(optimalTheta)

