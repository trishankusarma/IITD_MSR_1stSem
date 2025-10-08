#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from model import ModelClass


# In[2]:


dataX = np.loadtxt("q4x.dat")  # space separated
dataY = np.loadtxt("q4y.dat", dtype = str)

model = ModelClass()
X, Y = model.normalizeData(dataX, dataY)


# In[3]:


# Learn the Gaussian Discriminant Analysis parameters using the closed form equations described
# in class. Assume that both the classes have the same co-variance matrix i.e. Σ0 = Σ1 = Σ. Report the
# values of the means, µ0 and µ1, and the co-variance matrix Σ.
# phi0, phi1, mu0, mu1, sharedCovΣ = model.getParameters()
phi0, phi1, mu0, mu1, sharedCovΣ = model.getParameters()


# In[4]:


# Question 2 :: Plot the training data corresponding to the two coordinates of the input features, and you
# should use a different symbol for each point plotted to indicate whether that example had label Canada or
# Alaska.
model.plotTrainingData()


# In[5]:


# Question 3 :: Describe the equation of the boundary separating the two regions in terms of the parameters
# µ0, µ1 and Σ. Recall that GDA results in a linear separator when the two classes have identical co-variance
# matrix. Along with the data points plotted in the part above, plot (on the same figure) decision boundary
# fit by GDA.
W, b = model.getParametersForDecisionBoundaryForLinearGDA()
model.plotLinearGDA()


# In[6]:


# Question 4
# In general, GDA allows each of the target classes to have its own covariance matrix. This
# results (in general) results in a quadratic boundary separating the two class regions. In this case, the
# derive the maximum-likelihood estimate of the co-variance matrix Σ0

# And similarly, for Σ1. The expressions for the means remain the same as before. 
#Implement GDA for the above problem in this more general setting. Report the values of the parameter estimates i.e. µ0, µ1, Σ0,
# Σ1.
Σ0, Σ1 = model.getActualCovarianceMatrices()
P, Q, R = model.getQuadraticGDAParams()


# In[7]:


# Question 5
# Describe the equation for the quadratic boundary separating the two regions in terms of the
# parameters µ0, µ1 and Σ0, Σ1. On the graph plotted earlier displaying the data points and the linear
# separating boundary, also plot the quadratic boundary obtained in the previous step.
model.plot_quadratic_boundary()
model.plot_quadratic_boundary(addLinearBoundary = True)


# In[8]:





# In[ ]:




