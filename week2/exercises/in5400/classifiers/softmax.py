import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss=[]
  #dw = []
  # Using two tricks from the slides to avoid numerical instability in the softmax function:
  # 1. Shift exponential arguments max(z) to the right (in the sigmoid function)
  # 2. Take logarithm of modified sigmoid function from 1. and exponentiate it to get rid of division
  # Cross entropy loss for for a single sample is then computed by summing over all classes C
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    # Compute vector of scores. f_i.shape = (1,C)
    f_i = X[i].dot(W)

    # Normalization trick to avoid numerical instability, 
    # per http://cs231n.github.io/linear-classify/#softmax
    # Trick 1.
    f_i -= np.max(f_i)

    # Compute loss (and add to it, divided later).
    # Denominator of softmax function.
    sum_j = np.sum(np.exp(f_i))
    
    # Lambda function to compute log(softnmax) for sample i for all classes C
    # Numerator - denominator of sotmax function
    # Trick 2.
    log_p = lambda k: f_i[k] - np.log(sum_j)
    
    # The softmax for all classes C
    p = lambda k: np.exp(log_p(k))
    
    # Sum up crossentropy loss of the sample i
    # to become N samples minibatch crossentropy loss
    # for each class C
    loss += -np.log(p(y[i]))

    # Compute gradient
    # Here we are computing the contribution to the inner sum for a given sample i.
    for k in range(num_classes):
      p_k = p(k)
      # The gradient is (predicted class -  one hot encoded class) * image sample
      dW[:, k] += (p_k - (k == y[i])) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss = []
  #dW = []
  num_train = X.shape[0]
  
  # Compute matrix of scores
  f = X.dot(W)
  
  # Trick 1
  f -= np.max(f, axis=1, keepdims=True) # max of every sample
  
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
  
  # Trick 2
  log_p = f - np.log(sum_f)
    
  #p = np.exp(f)/sum_f

  loss = np.sum(-log_p[np.arange(num_train), y])

  ind = np.zeros_like(log_p)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(np.exp(log_p) - ind)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

