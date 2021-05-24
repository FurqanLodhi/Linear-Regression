#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment 2: Linear Regression
# 
# ## Instructions
# 
# - The aim of this assignment is to give you a hands-on with a real-life machine learning application.
# - Use separate training, and testing data as discussed in class.
# - You can only use Python programming language and Jupyter Notebooks.
# - There are three parts of this assignment. In parts 1 & 2, you can only use **numpy, scipy, pandas, matplotlib and are not allowed to use NLTK, scikit-learn or any other machine learning toolkit**. However, you have to use **scikit-learn** in part 3.
# - Carefully read the submission instructions, plagiarism and late days policy below.
# - Deadline to submit this assignment is: **Friday, 30th October 2020**.
# 
# ## Submission Instructions
# 
# Submit your code both as notebook file (.ipynb) and python script (.py) on LMS. The name of both files should be your roll number. If you don’t know how to save .ipynb as .py [see this](https://i.stack.imgur.com/L1rQH.png). **Failing to submit any one of them will result in the reduction of marks**.
# 
# ## Plagiarism Policy
# 
# The code MUST be done independently. Any plagiarism or cheating of work from others or the internet will be immediately referred to the DC. If you are confused about what constitutes plagiarism, it is YOUR responsibility to consult with the instructor or the TA in a timely manner. No “after the fact” negotiations will be possible. The only way to guarantee that you do not lose marks is “DO NOT LOOK AT ANYONE ELSE'S CODE NOR DISCUSS IT WITH THEM”.
# 
# ## Late Days Policy
# 
# The deadline of the assignment is final. However, in order to accommodate all the 11th hour issues there is a late submission policy i.e. you can submit your assignment within 3 days after the deadline with 25% deduction each day.
# 
# 
# ## Introduction
# 
# In this exercise, you will implement linear regression and get to see it work on data. After completing this assignment, you will know:
# - How to implement linear regression from scratch.
# - How to estimate linear regression parameters using gradient descent.
# - How to make predictions on new data using learned parameters.
# 
# Let's start with the necessary imports.

# In[1]:


import os
import numpy as np
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Linear Regression with one variable
# 
# Linear regression assumes a linear relationship between the input variables (X) and the single output variable (Y). More specifically, that output (Y) can be calculated from a linear combination of the input variables (X). When there is a single input variable, the method is referred to as a simple linear regression.
# 
# Now you will implement simple linear regression to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

# ### 1.1 Load the dataset
# 
# The file `Data/ex1data1.txt` contains the dataset for our linear regression problem. The first column is the population of a city (in 10,000s) and the second column is the profit of a food truck in that city (in $10,000s). A negative value for profit indicates a loss. 
# 
# We provide you with the code needed to load this data. The dataset is loaded from the data file into the variables `X` and `Y`.

# In[2]:


data = np.loadtxt(os.path.join('Data', 'ex1data.txt'), delimiter=',')
X, Y = data[:, 0], data[:, 1]


# ### 1.2 Plot the dataset
# Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). Execute the next cell to visualize the data.

# In[3]:


pyplot.plot(X, Y, 'ro', ms=10, mec='k')
pyplot.ylabel('Profit in $10,000')
pyplot.xlabel('Population of City in 10,000s')


# ### 1.3 Learn the parameters
# In this part, you will fit the linear regression parameters $\theta$ to the food truck dataset using gradient descent.
# 
# The objective of linear regression is to minimize the cost function
# 
# $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_{\theta}(x^{(i)}) - y^{(i)}\right)^2 ------ (i)$$ 
# 
# where the hypothesis $h_\theta(x)$ is given by the linear model
# $$ h_\theta(x) = \theta_0 + \theta_1 x ------ (ii)$$
# 
# The parameters of your model are the $\theta_j$ values. These are
# the values you will adjust to minimize cost $J(\theta)$. One way to do this is to
# use the batch gradient descent algorithm. In batch gradient descent, each
# iteration performs the update
# 
# $$ \theta_0 = \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)}\right) ------ (iii)$$
# 
# $$ \theta_1 = \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)}\right)x^{(i)} ------ (iv)$$
# 
# With each step of gradient descent, your parameters $\theta_j$ come closer to the optimal values that will achieve the lowest cost J($\theta$).
# 
# Let's start by implementing the hypothesis $h_\theta(x)$.

# In[4]:


### GRADED FUNCTION ###
def predict(x, theta0, theta1):
    '''
    Calculates the hypothesis for any input sample `x` given the parameters `theta`.
    
    Arguments
    ---------
    x : float
        The input sample.
    
    theta0 : float
        The parameter for the regression function.
        
    theta1 : float
        The parameter for the regression function.
    
    Returns
    -------
    h_x : float
        The hypothesis for input sample.
    
    Hint(s)
    -------
    Compute equation (ii).
    '''
    # You need to return the following variable(s) correctly
    h_x = 0.0
    
    ### START CODE HERE ### (≈ 1 line of code)
    h_x = theta0 + theta1*x
    ### END CODE HERE ###
    
    return h_x


# Execute the next cell to verify your implementation.

# In[5]:


h_x = predict(x=2, theta0=1.0, theta1=1.0)
print('With x = 2, theta0 = 1.0, theta1 = 1.0\nPredicted Hypothesis h(x) = %.2f' % h_x)
print("Expected hypothesis h(x) = 3.00\n")


# As you perform gradient descent to learn minimize the cost function  $J(\theta)$, it is helpful to monitor the convergence by computing the cost. In this section, you will implement a function to calculate  $J(\theta)$ so you can check the convergence of your gradient descent implementation.

# In[6]:


### GRADED FUNCTION ###
def computeCost(X, Y, theta0, theta1):
    '''
    Computes cost for linear regression. Computes the cost of using `theta` as the
    parameter for linear regression to fit the data points in `X` and `Y`.
    
    Arguments
    ---------
    X : array
        The input dataset of shape (m, ), where m is the number of training examples.
    
    Y : array
        The values of the function at each data point. This is a vector of
        shape (m, ), where m is the number of training examples.
    
    theta0 : float
        The parameter for the regression function.
        
    theta1 : float
        The parameter for the regression function.
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Hint(s)
    -------
    Compute equation (i).
    '''
    # initialize some useful values
    m = Y.size  # number of training examples
    
    # You need to return the following variable(s) correctly
    J = 0
        
    ### START CODE HERE ### (≈ 3-4 lines of code)
    
    h_x = predict(X, theta0, theta1)   
        
    J = 1/(2*m) * np.sum((h_x-Y)**2)  
        
    ### END CODE HERE ###
    
    return J


# Execute the next cell to verify your implementation.

# In[7]:


J = computeCost(X, Y, theta0=1.0, theta1=1.0)
print('With theta0 = 1.0, theta1 = 1.0\nPredicted cost J = %.2f' % J)
print("Expected cost J = 10.27\n")


# Next, you will complete a function which implements gradient descent. The loop structure has been written for you, and you only need to supply the updates to parameters $\theta_j$  within each iteration (epoch). 
# 
# The starter code for the function `gradientDescent` calls `computeCost` on every iteration and saves the cost to a `python` list. Assuming you have implemented `gradientDescent` and `computeCost` correctly, your value of $J(\theta)$ should never increase, and should converge to a steady value by the end of the algorithm.

# In[8]:


### GRADED FUNCTION ###
def gradientDescent(X, Y, alpha, n_epoch):
    """
    Performs gradient descent to learn `theta`. Updates `theta` by taking `n_epoch`
    gradient steps with learning rate `alpha`.
    
    Arguments
    ---------
    X : array
        The input dataset of shape (m, ), where m is the number of training examples.
    
    Y : array
        The values of the function at each data point. This is a vector of
        shape (m, ), where m is the number of training examples.
    
    alpha : float
        The learning rate.
    
    n_epoch : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta0 : float
        The parameter for the regression function.
        
    theta1 : float
        The parameter for the regression function.
    
    J : list
        A python list for the values of the cost function after each iteration.
    
    Hint(s)
    -------
    Compute equation (iii) and (iv).

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) here.
    """
    # initialize some useful values
    m = Y.size  # number of training examples
    J = list()  # list to store cost
    
    # You need to return the following variables correctly
    theta0 = 0.0
    theta1 = 0.0
        
    for epoch in range(n_epoch):
        ### START CODE HERE ### (≈ 5-10 lines of code)
        
        h_x = predict(X, theta0, theta1)
        
        theta0 = theta0 - alpha*1/m * np.sum(h_x-Y)
        theta1 = theta1 - alpha*1/m * np.sum((h_x-Y)*X) 
        
        ### END CODE HERE ###

        J.append(computeCost(X, Y, theta0, theta1))
    return theta0, theta1, J


# Execute the next cell to verify your implementation.

# In[9]:


n_epoch = 1500
alpha = 0.01

theta0, theta1, J = gradientDescent(X ,Y, alpha, n_epoch)
print('Predicted theta0 = %.4f, theta1 = %.4f, cost = %.4f' % (theta0, theta1, J[-1]))
print('Expected theta0 = -3.6303, theta1 = 1.1664, cost = 4.4834')


# ### 1.4 Plot the linear fit
# 
# Use your learned parameters $\theta_j$ to plot the linear fit.

# In[10]:


h_x = list()
for x in X:
    h_x.append(predict(x, theta0, theta1))
pyplot.plot(X, Y, 'ro', ms=10, mec='k')
pyplot.ylabel('Profit in $10,000')
pyplot.xlabel('Population of City in 10,000s')
pyplot.plot(X, h_x, '-')
pyplot.legend(['Training data', 'Linear regression'])


# ### 1.5 Make predictions
# 
# Use your learned parameters $\theta_j$ to make food truck profit predictions in areas with population of 40,000 and 65,000.

# In[11]:


print('For population = 40,000, predicted profit = $%.2f' % (predict(4, theta0, theta1)*10000))
print('For population = 65,000, predicted profit = $%.2f' % (predict(6.5, theta0, theta1)*10000))


# ## 2. Multivariate Linear Regression
# 
# Now, you will implement multivariate linear regression (from scratch) to predict the the median price of homes in a Boston suburb during the mid-1970s. To do this, you are given with the dataset that has 404 examples in the train set and 102 examples in test set. Each example has 13 input variables (features) and one output variable (price in $10,000s). Below is the description of input variables:
# 
# - Per capita crime rate.
# - The proportion of residential land zoned for lots over 25,000 square feet.
# - The proportion of non-retail business acres per town.
# - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# - Nitric oxides concentration (parts per 10 million).
# - The average number of rooms per dwelling.
# - The proportion of owner-occupied units built before 1940.
# - Weighted distances to five Boston employment centers.
# - Index of accessibility to radial highways.
# - Full-value property-tax rate per $10,000.
# - Pupil-teacher ratio by town.
# - 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
# - Percentage lower status of the population.
# 
# Each one of these input features is stored using a different scale. Some features are represented by a proportion between 0 and 1, other features are ranges between 1 and 12, some are ranges between 0 and 100, and so on. This is often the case with real-world data, and understanding how to explore and clean such data is an important skill to develop.
# 
# A common way to normalize features that use different scales and ranges is:
# 
# - Subtract the mean value of each feature from the dataset.
# - After subtracting the mean, additionally scale (divide) the feature values by their respective standard deviations.
# 
# Note: We only use examples of the train set to estimate the mean and standard deviation.
# 
# You have to follow exactly the same steps as above i.e. implement hypothesis, cost function and gradient descent for multivariate linear regression to learn parameters $\theta$ using train set. Finally, report the cost (error) using your learned parameters $\theta$ on test set. Expected Mean Square Error on this dataset is 11.5 - 12.5 approximately. 
# 
# We provide you with the code needed to load this dataset. The dataset is loaded from the data files into the variables `train_X`, `train_Y`, `test_X` and `test_Y`.

# In[12]:


train_X = np.loadtxt(os.path.join('Data', 'ex2traindata.txt'))
train_Y = np.loadtxt(os.path.join('Data', 'ex2trainlabels.txt'))
test_X = np.loadtxt(os.path.join('Data', 'ex2testdata.txt'))
test_Y = np.loadtxt(os.path.join('Data', 'ex2testlabels.txt'))


# In[13]:


''' Start your code of part 2 from here, add the new code cells as per your requirement. '''

mean = np.mean(train_X, axis = 0)
standDev = np.std(train_X, axis = 0)

normalizedTrain_X = (train_X - mean)/standDev
normalizedTest_X = (test_X - mean)/standDev

# print(normalizedTrain_X)

biasesTrain = np.ones((normalizedTrain_X.shape[0],1))
biasesTest = np.ones((normalizedTest_X.shape[0],1))

NTrainX = np.append(biasesTrain, normalizedTrain_X, axis = 1)
NTestX = np.append(biasesTest, normalizedTest_X, axis = 1)

train_Y = np.reshape(train_Y, (404,1))
test_Y = np.reshape(test_Y, (102,1))


# In[14]:


thetaList = np.zeros((NTrainX.shape[1], 1))

# print(NTrainX.shape)
# print(thetaList.shape)
# print(train_Y.shape)


# In[15]:


def predictMulti(X, thetas):
    
    h_x = np.dot(X, thetas)
    return h_x


# In[16]:


def computeCostMulti(X, Y, thetas):
    
    m = Y.size
    h_x = predictMulti(X, thetas)    
    J = 1/(2*m) * np.sum((h_x-Y)**2)    
    return J


# In[17]:


def gradDescentMulti(X, Y, alpha, n_epoch, thetas):
    
    m = Y.size
    J = list()
    
    
#     thetaList = np.zeros((X.shape[1], 1))
#     print("Thetas: ",thetas.shape)
    for epoch in range(n_epoch):
#         print("H_X", h_x.shape)
#         print("Y", Y.shape)
#         print("X",X.shape)
#         print("h_x - Y", np.subtract(h_x, Y).shape)
        for i in range(X.shape[0]):
            h_x = predictMulti(X[i], thetas) 
            for j in range(X.shape[1]):
                thetas[j] = thetas[j] - alpha*1/m * np.dot(X[i][j], np.subtract(h_x, Y[i]))
        J.append(computeCostMulti(X, Y, thetas))
#     print(thetas.shape)
#     print(thetas)
    return thetas, J


# In[18]:


# print(thetaList)

thetaList, J = gradDescentMulti(NTrainX, train_Y, alpha, n_epoch, thetaList)

# print(thetaList.shape)


# In[19]:


def testing(X, Y, thetaList):
    h_x = predictMulti(X, thetaList)
#     print(h_x.shape)
    print("Mean Squared Error: ", computeCostMulti(X, Y, thetaList))


# In[20]:


testing(NTestX, test_Y, thetaList)


# ## 3. Regularized Linear Regression
# 
# Now, you'll use the [scikit-learn](https://scikit-learn.org/stable/index.html) to implement [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso), [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet) and apply them to Boston house pricing dataset (provided in part 2). Try out different values of regularization coefficient (known as alpha in scikit-learn) and use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to report loss with each regression. Finally, plot the regularization coefficients alpha (x-axis) with learned parameters $\theta$ (y-axis) for Ridge and Lasso. Please read [this blog](https://scienceloft.com/technical/understanding-lasso-and-ridge-regression/) to get better understanding of the desired plots.

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


# In[22]:


''' Start your code of part 3 from here, add the new code cells as per your requirement. '''


# In[23]:


linReg = LinearRegression()

linReg.fit(NTrainX, train_Y)
h_x = linReg.predict(NTestX)
MSELinReg = mean_squared_error(h_x, test_Y)

print("Mean Squared Error: ", MSELinReg)


# In[24]:


rdge = Ridge(alpha = 100)

rdge.fit(NTrainX, train_Y)
h_x = rdge.predict(NTestX)
MSERidge = mean_squared_error(h_x, test_Y)

print("Mean Squared Error: ", MSERidge)


# In[25]:


lsso = Lasso(alpha = 0.2)

lsso.fit(NTrainX, train_Y)
h_x = lsso.predict(NTestX)
MSELasso = mean_squared_error(h_x, test_Y)

print("Mean Squared Error: ", MSELasso)


# In[26]:


elasNet = ElasticNet(alpha = 0.2)
elasNet.fit(NTrainX, train_Y)
h_x = elasNet.predict(NTestX)
MSEElasNet = mean_squared_error(h_x, test_Y)

print("Mean Squared Error: ", MSEElasNet)


# In[27]:


alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000, 10000, 100000]

for alpha in alphas:
    rdge = Ridge(alpha = alpha)

    rdge.fit(NTrainX, train_Y)
    h_x = rdge.predict(NTestX)
    MSERidge = mean_squared_error(h_x, test_Y)

    print("Mean Squared Error Ridge Regression for alpha: ", alpha, "is", MSERidge)


# In[28]:


for alpha in alphas:
    lsso = Lasso(alpha = alpha)

    lsso.fit(NTrainX, train_Y)
    h_x = lsso.predict(NTestX)
    MSELasso = mean_squared_error(h_x, test_Y)

    print("Mean Squared Error Lasso Regression for alpha: ", alpha, "is", MSELasso)


# In[29]:


for alpha in alphas:
    elasNet = ElasticNet(alpha = alpha)
    elasNet.fit(NTrainX, train_Y)
    h_x = elasNet.predict(NTestX)
    MSEElasNet = mean_squared_error(h_x, test_Y)

    print("Mean Squared Error Elastic Net Regression for alpha: ", alpha, "is", MSEElasNet)


# In[30]:


labels = [
    "Per capita crime rate",
    "residential land zoned",
    "business acres per town.",
    "Charles River dummy variable",
    "Nitric oxides concentration",
    "average number of rooms",
    "owner-occupied units",
    "distances to Boston employment centers.",
    "accessibility to radial highways.",
    "property-tax rate",
    "Pupil-teacher ratio",
    "1000 * (Bk - 0.63) ** 2",
    "Percentage lower status of the population."
]


# In[31]:


coeffLasso = []

for alpha in alphas:
    lsso = Lasso(alpha = alpha)
    lsso.fit(NTrainX, train_Y)
    coeffLasso.append(lsso.coef_)

# print(coeffLasso)


# In[32]:


pyplot.plot(np.log10(alphas), coeffLasso)
pyplot.xlabel("log Alphas")
pyplot.ylabel("Coefficients")
pyplot.title("Lasso Regularization")
pyplot.legend(labels, bbox_to_anchor=(2, 1))


# In[33]:


coeffRidge = []

for alpha in alphas:
    rdge = Ridge(alpha = alpha)
    rdge.fit(NTrainX, train_Y)
    coeffRidge.append(rdge.coef_)

coeffRidge = [e for sl in coeffRidge for e in sl]


# In[34]:


pyplot.plot(np.log10(alphas), coeffRidge)
pyplot.xlabel("log Alphas")
pyplot.ylabel("Coefficients")
pyplot.title("Ridge Regularization")
pyplot.legend(labels, bbox_to_anchor=(2, 1))


# In[ ]:





# In[ ]:




