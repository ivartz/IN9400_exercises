#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    layer_dimensions = conf["layer_dimensions"]
    params = \
    { "b_" + str(i+1) : np.zeros(shape=(l_size, 1)) for i, l_size in enumerate(layer_dimensions[1:])}
    params.update({ "W_" + str(i+1) : \
                   np.random.normal(scale=np.sqrt(2/layer_dimensions[i]), \
                                    size=(layer_dimensions[i],l_size)) \
                   for i, l_size in enumerate(layer_dimensions[1:])})
    
    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        Z[Z<0] = 0
        return Z
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 b)
    # Trick 1
    Z -= np.max(Z, axis=0, keepdims=True) # max of every sample (m samples)

    sum_Z = np.sum(np.exp(Z), axis=0, keepdims=True)

    # Trick 2
    log_softmax = Z - np.log(sum_Z)
    return np.exp(log_softmax)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)
    features = dict()
    
    layer_dimensions = conf["layer_dimensions"]
    num_layers = len(layer_dimensions)
    
    n_x, m = X_batch.shape
    
    for l in range(num_layers):
        if l == 0:
            # At the input layer
            # Activations are the images in the batches
            A = X_batch
            if is_training:
                features["A_0"] = A
        else:
            # The vectorized equations are explained in the slices 
            # week3/lecture_material/in5400_lectures_week03_slides.pdf at page 61.
            # Vectorized equations for computing the linear combinations z = aw + b
            # and activation(z) for a batch consisting of several samples (images).
                        
            #Z = np.dot(params["W_" + str(l)].T, A) + \
            #params["b_" + str(l)]*np.ones(shape=(layer_dimensions[l], m))
            
            # B will be broadcasted to the m samples.
            Z = np.dot(params["W_" + str(l)].T, A) + params["b_" + str(l)]
            
            if l + 1 == num_layers:
                # At the output layer.
                # Compute softmax output.
                Y_proposed = softmax(Z)
                if is_training:
                    # Store linear combinations
                    # on output neurons for
                    # later use in backprogation.
                    features["Z_" + str(l)] = Z
                    features["A_" + str(l)] = Y_proposed
            else:
                # In a hidden layer.
                # Compute activation.
                A = activation(Z, activation_function="relu")
                if is_training:
                    # Store the linear combinations
                    # and the activations for later 
                    # use in backward propagation.
                    features["Z_" + str(l)] = Z
                    features["A_" + str(l)] = A
    
    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    # Referring to week3/lecture_material/in5400_lectures_week03_slides.pdf at page 62
    # for vectorized implementation of cross entropy cost function.
    
    n_y, m = Y_reference.shape
    
    cost = -(1/m)*np.dot(np.dot(np.ones(shape=(1,n_y)),Y_reference*np.log(Y_proposed)),np.ones(shape=(m,1)))
    
    max_probability_indices = np.vstack((np.argmax(Y_proposed, axis=0), np.arange(m)))
    
    num_correct = np.sum(Y_reference[max_probability_indices[0], max_probability_indices[1]])

    return np.float(cost), num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        # Using Heaviside step-function
        Z[Z >= 0] = 1
        Z[Z < 0 ] = 0
        return Z
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    layer_dimensions = conf["layer_dimensions"]
    num_layers = len(layer_dimensions)
    
    n_y, m = Y_reference.shape

    grad_params = dict()
    
    for l in reversed(range(1, num_layers)):
        if l+1 == num_layers:
            # l is the output layer.
            # Computing the Jacobian 
            # of the last layer
            # J_z = Y_proposed - Y_reference
            # Y_reference are one-hot encoded true class labels.
            # J_z is a special case when
            # using cross entropy on output
            # layer with softmax activation
            # (on output layer neurons).
            Jac_z = Y_proposed - Y_reference
        else:
            # l is a hidden layer.
            # General Jacobian derived for 
            # any activation function.
            Jac_z = activation_derivative(features["Z_" + str(l)], activation_function="relu")*\
            np.dot(params["W_" + str(l+1)], Jac_z)

        grad_params["grad_W_" + str(l)] = (1/m)*np.dot(features["A_" + str(l-1)], Jac_z.T)
        grad_params["grad_b_" + str(l)] = (1/m)*np.dot(Jac_z, np.ones(shape=(m, 1)))
    
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    updated_params = dict()
    
    for k, v in params.items():
        updated_params[k] = v - conf["learning_rate"]*grad_params["grad_" + k]
        
    return updated_params
