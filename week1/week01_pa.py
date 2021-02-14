
# coding: utf-8

# # Programming assignment (Linear models, Optimization)
# 
# In this programming assignment you will implement a linear classifier and train it using stochastic gradient descent modifications and numpy.

# In[11]:


import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[12]:


import sys
sys.path.append("..")
import grading
grader = grading.Grader(assignment_key="UaHtvpEFEee0XQ6wjK-hZg", 
                      all_parts=["xU7U4", "HyTF6", "uNidL", "ToK7N", "GBdgZ", "dLdHG"])


# In[13]:


grader


# In[263]:


# token expires every 30 min
COURSERA_TOKEN = 'Ns5eUh8JKe3fEPKN'
COURSERA_EMAIL = 'samuel.herilces@alu.ufc.br'


# ## Two-dimensional classification
# 
# To make things more intuitive, let's solve a 2D classification problem with synthetic data.

# In[15]:


with open('train.npy', 'rb') as fin:
    X = np.load(fin)
    
with open('target.npy', 'rb') as fin:
    y = np.load(fin)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
plt.show()


# # Task
# 
# ## Features
# 
# As you can notice the data above isn't linearly separable. Since that we should add features (or use non-linear model). Note that decision line between two classes have form of circle, since that we can add quadratic features to make the problem linearly separable. The idea under this displayed on image below:
# 
# ![](kernel.png)

# In[16]:


def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
        
    # TODO:<your code here>
    X_expanded[:,0] = X[:,0]
    X_expanded[:,1] = X[:,1]
    X_expanded[:,2] = X[:,0]**2
    X_expanded[:,3] = X[:,1]**2
    X_expanded[:,4] = X[:,0]*X[:,1]
    X_expanded[:,5] = [1 for i in range(X.shape[0])]
    
    return X_expanded


# In[17]:


X_expanded = expand(X)


# Here are some tests for your implementation of `expand` function.

# In[18]:


# simple test on random numbers

dummy_X = np.array([
        [0,0],
        [1,0],
        [2.61,-1.28],
        [-0.59,2.1]
    ])

# call your expand function
dummy_expanded = expand(dummy_X)

# what it should have returned:   x0       x1       x0^2     x1^2     x0*x1    1
dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],
                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],
                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],
                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])

#tests
assert isinstance(dummy_expanded,np.ndarray), "please make sure you return numpy array"
assert dummy_expanded.shape == dummy_expanded_ans.shape, "please make sure your shape is correct"
assert np.allclose(dummy_expanded,dummy_expanded_ans,1e-3), "Something's out of order with features"

print("Seems legit!")


# ## Logistic regression
# 
# To classify objects we will obtain probability of object belongs to class '1'. To predict probability we will use output of linear model and logistic function:
# 
# $$ a(x; w) = \langle w, x \rangle $$
# $$ P( y=1 \; \big| \; x, \, w) = \dfrac{1}{1 + \exp(- \langle w, x \rangle)} = \sigma(\langle w, x \rangle)$$
# 

# In[19]:


def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """

    # TODO:<your code here>
    exp = -1*(expand(X)@w)
    return 1/(1+np.power(np.e,exp))


# In[20]:


dummy_weights = np.linspace(-1, 1, 6)
ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]


# In[21]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("xU7U4", ans_part1)


# In[22]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# In logistic regression the optimal parameters $w$ are found by cross-entropy minimization:
# 
# Loss for one sample: $$ l(x_i, y_i, w) = - \left[ {y_i \cdot log P(y_i = 1 \, | \, x_i,w) + (1-y_i) \cdot log (1-P(y_i = 1\, | \, x_i,w))}\right] $$
# 
# Loss for many samples: $$ L(X, \vec{y}, w) =  {1 \over \ell} \sum_{i=1}^\ell l(x_i, y_i, w) $$
# 
# 

# In[23]:


def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function L using formula above.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    # TODO:<your code here>
    part_a = y*np.log(probability(X, w))
    part_b = ((1-y)*np.log(1-probability(X, w)))
    result = part_a+part_b
    return np.sum(-result)/X.shape[0]


# In[24]:


# use output of this cell to fill answer field 
ans_part2 = compute_loss(X_expanded, y, dummy_weights)
ans_part2


# In[25]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("HyTF6", ans_part2)


# In[26]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# Since we train our model with gradient descent, we should compute gradients.
# 
# To be specific, we need a derivative of loss function over each weight [6 of them].
# 
# $$ \nabla_w L = {1 \over \ell} \sum_{i=1}^\ell \nabla_w l(x_i, y_i, w) $$ 
# 
# We won't be giving you the exact formula this time — instead, try figuring out a derivative with pen and paper. 
# 
# As usual, we've made a small test for you, but if you need more, feel free to check your math against finite differences (estimate how $L$ changes if you shift $w$ by $10^{-5}$ or so).

# In[27]:


def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    
    # TODO<your code here>   
    error = y - probability(X, w)
    derivate = [np.dot(X[:,i],error) for i in range(6)]
    return np.dot(derivate,1/X.shape[0])


# In[28]:


# use output of this cell to fill answer field 
ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))
ans_part3


# In[29]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("uNidL", ans_part3)


# In[30]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# Here's an auxiliary function that visualizes the predictions:

# In[31]:


from IPython import display

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()


# In[32]:


visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])


# ## Training
# In this section we'll use the functions you wrote to train our classifier using stochastic gradient descent.
# 
# You can try change hyperparameters like batch size, learning rate and so on to find the best one, but use our hyperparameters when fill answers.

# ## Mini-batch SGD
# 
# Stochastic gradient descent just takes a random batch of $m$ samples on each iteration, calculates a gradient of the loss on it and makes a step:
# $$ w_t = w_{t-1} - \eta \dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t) $$
# 
# 

# In[166]:


# please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # Keep in mind that compute_grad already does averaging over batch for you!
    # TODO:<your code here>
    grad = compute_grad(X_expanded[ind,:], y[ind], w)
    w = w + np.dot(eta,grad)

visualize(X, y, w, loss)
plt.clf()


# In[167]:


# use output of this cell to fill answer field 

ans_part4 = compute_loss(X_expanded, y, w)
ans_part4


# In[168]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("ToK7N", ans_part4)


# In[169]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# ## SGD with momentum
# 
# Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in image below. It does this by adding a fraction $\alpha$ of the update vector of the past time step to the current update vector.
# <br>
# <br>
# 
# $$ \nu_t = \alpha \nu_{t-1} + \eta\dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t) $$
# $$ w_t = w_{t-1} - \nu_t$$
# 
# <br>
# 
# 
# ![](sgd.png)
# 

# In[181]:


# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)
v = 0

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    grad = compute_grad(X_expanded[ind,:], y[ind], w)
    v = alpha*v + np.dot(eta,grad)
    w = w + v
    
visualize(X, y, w, loss)
plt.clf()


# In[182]:


# use output of this cell to fill answer field 

ans_part5 = compute_loss(X_expanded, y, w)
ans_part5


# In[183]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("GBdgZ", ans_part5)


# In[184]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# ## RMSprop
# 
# Implement RMSPROP algorithm, which use squared gradients to adjust learning rate:
# 
# $$ G_j^t = \alpha G_j^{t-1} + (1 - \alpha) g_{tj}^2 $$
# $$ w_j^t = w_j^{t-1} - \dfrac{\eta}{\sqrt{G_j^t + \varepsilon}} g_{tj} $$

# In[267]:


# please use np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1.])

eta = 0.1 # learning rate
alpha = 0.9 # moving average of gradient norm squared
g2 = 0 # we start with None so that you can update this value correctly on the first iteration
eps = 1e-8

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    g = compute_grad(X_expanded[ind,:], y[ind], w)
    
    g2 = alpha*g2 + np.dot(1-alpha,g*g)    
    w = w + (eta/np.sqrt(g2+eps))*g
    
    
visualize(X, y, w, loss)
plt.clf()


# In[268]:


# use output of this cell to fill answer field 
ans_part6 = compute_loss(X_expanded, y, w)
ans_part6


# In[269]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("dLdHG", ans_part6)


# In[270]:


grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

