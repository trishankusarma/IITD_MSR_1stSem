#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import ModelClass

# these command help to have plots inline with output cells
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dfX = pd.read_csv('./linearX.csv', header = None, names=['X'])
dfY = pd.read_csv('./linearY.csv', header = None, names=['Y'])

print("X.shape : ",dfX.shape)
print("Y.shape : ",dfY.shape)

num_points = dfX.shape[0]
print("Number of data points : ",num_points)

num_features = dfX.shape[1]
print("Number of features : ",num_features)


# In[3]:


# #Plot the graph over x-y plane
# sns.scatterplot(x = dfX.values[:, 0], y = dfY.values[:,0], color='red')
# plt.title('Actual X v/s ground truth')
# plt.ylabel('Y')
# plt.xlabel('X')
# plt.show()


# In[4]:


model = ModelClass()
X, Y, theta = model.initializeXYAndParams(dfX, dfY)


# In[5]:


# Q1 :: implement batch gradient descent and get the final set of parameters learnt by the algorithm
# So,
# choosing a learning_rate of 0.01
# Stopping criteria :: ((epocs+1) >= total_epocs) or (epocs > 1 and (lossPrev - lossCurr) < tolerance)
# maxm_epocs = 100000
# convergence_threshold = 1e-6
# report the final set of parameters
learning_rate = 0.01
total_epocs = 100000
tolerance = 1e-6


# In[6]:


theta_predicted, loss, params = model.runBatchGD(theta, learning_rate = learning_rate, total_epocs = total_epocs, tolerance = tolerance)
sns.scatterplot(x = range(0,len(loss)), y = loss, color='red')

plt.plot(loss, color='red')
plt.title('Loss vs Epocs')
plt.xlabel('Epocs')
plt.ylabel('Loss')
plt.show()

# learning_rate = 0.001 total_epocs = 100000 tolerance = 1e-6
# Epoch 6849, Loss: 0.005374383131988457
# Theta : [[ 6.2121121 ]
#  [29.03385654]]

# 1e-4
# Epoch 4546, Loss: 0.05475500453543195
# Theta : [[ 6.15291603]
#  [28.75599701]]

# learning_rate = 0.01 , tolerance = 1e-6
# Epoch 798, Loss: 0.004923384323470479
# Theta : [[ 6.21665396]
#  [29.05521858]]


# In[7]:


# Part 2 :: plot of data and hypothesis function learnt
# On the parameters learnt predicting the graph
Y_predict = np.dot(theta_predicted.T, X)

sns.scatterplot(x = dfX.values[:, 0], y =  dfY.values[:,0], color='blue')
plt.title('Actual X v/s ground truth')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

#Plot the predicted graph over x-y plane
sns.scatterplot(x = dfX.values[:, 0], y = Y_predict.T[:,0], color='red')
plt.title('Actual X v/s predicted value')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


# In[8]:


# Question 3A and 3B :: for plotting the 3D mesh grid and displaying the error value 
model.get3DGradientDescentPlot(params, learning_rate = learning_rate, iterations = total_epocs, tolerance = tolerance, sleepSeconds = 0.02)


# In[9]:


# 4 for plotting the contour and displaying the error value
model.getContourPlot(params, learning_rate = learning_rate, iterations = total_epocs, tolerance = tolerance, sleepSeconds = 0.002)


# In[9]:


# Question 5:: repeat the above contour plots for different learning rates
learning_rates = [0.001, 0.025, 0.1]

for learning_rate in learning_rates:
    print("Current Learning rate : ",learning_rate)
    model.getContourPlot(params, learning_rate = learning_rate, iterations = 10000, tolerance = 1e-4, sleepSeconds = 0.002)


# In[ ]:




