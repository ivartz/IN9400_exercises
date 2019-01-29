import numpy as np
#from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    

    dists = self.compute_distances(X)
    
    return self.predict_labels(dists, k=k)

  

  
  

  def compute_distances(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train.


    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    #                                     #
    #########################################################################
    
    """
    # Almost slowest possible method
    
    def l2_norm(img1, img2):
        # Assume img1 and img2 are 1D vectors
        sum = 0
        for pixel1 in img1:
            for pixel2 in img2:
                sum += (pixel1 - pixel2)**2
        return np.sqrt(sum)
    
    for test_image_index in range(num_test):
        for train_image_index in range(num_train):
            
            dists[test_image_index, train_image_index] = \
                l2_norm(X[test_image_index], self.X_train[train_image_index])
    """
    """
    # Faster
    for test_image_index in range(num_test):
        for train_image_index in range(num_train):
            
            dists[test_image_index, train_image_index] = \
                np.sqrt( np.sum( (X[test_image_index] - self.X_train[train_image_index])**2 ) )
    """
    #"""
    # Fastest
    # split (p-q)^2 to p^2 + q^2 - 2pq
    dists = np.sqrt( (X**2).sum(axis=1, keepdims=True) + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T) )
    #"""
    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    closest_y = np.zeros((1,k),dtype='int32')
    for i in range(0,num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        #closest_y = []
        #########################################################################
        # TODO:                                                                 #
        # Use the distance matrix to find the k nearest neighbors of the ith    #
        # testing point, and use self.y_train to find the labels of these       #
        # neighbors. Store these labels in closest_y.                           #
        # Hint: Look up the function numpy.argsort.                             #
        #########################################################################
        
        """
        # My method
        for neighbor in range(k):
            
            closest_y[0][neighbor] = \
                self.y_train[np.argwhere(dists[i] == np.sort(dists[i])[neighbor])]
        """
        #"""
        # Cleaner method
        closest_y[0] = self.y_train[np.argsort(dists[i])][0:k]
        #"""
        
        #########################################################################
        # TODO:                                                                 #
        # Now that you have found the labels of the k nearest neighbors, you    #
        # need to find the most common label in the list closest_y of labels.   #
        # Store this label in y_pred[i]. Break ties by choosing the smaller     #
        # label.                                                                #
        #########################################################################

        counts = np.bincount(closest_y[0])
        y_pred[i] = np.argmax(counts)

        #########################################################################
        #                           END OF YOUR CODE                            # 
        #########################################################################

    return y_pred



