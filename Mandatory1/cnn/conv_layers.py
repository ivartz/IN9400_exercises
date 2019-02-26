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

"""Implementation of convolution forward and backward pass"""

import numpy as np
import sys

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1   
    def compute_output_layer_shape(input_heigth, \
                                   input_width, \
                                   kernel_heigth, \
                                   kernel_width, \
                                   pad_size, \
                                   stride):
        # Formula for the spatial dimensions of an image
        # in an convolutional output layer.
        return np.int(np.floor(1 + (input_heigth + 2*pad_size - kernel_heigth)/stride)), \
                np.int(np.floor(1 + (input_width + 2*pad_size - kernel_width)/stride))
    
    # Flipping kernels to so that it is convolution and not correlation.
    # This is not necessary, since the weights are random initialized.
    #weight = weight[:,:,::-1,::-1]

    num_filters, _, height_weights, width_weights = weight.shape
    
    batch_size, channels_input_layer, height_input_layer, width_input_layer = \
                                                    input_layer.shape

    # Define dimension of output layer.
    height_output_layer, width_output_layer,  = \
                    compute_output_layer_shape(height_input_layer, \
                                                 width_input_layer, \
                                                 height_weights, \
                                                 width_weights, \
                                                 pad_size, \
                                                 stride)
    channels_output_layer = num_filters

    # Should have shape (batch_size, num_filters, height_y, width_y)
    output_layer = np.zeros(shape=(batch_size, \
                                   channels_output_layer, \
                                   height_output_layer, \
                                   width_output_layer))

    # The center of the kernel is directly above the output
    # pixel. The coordinates for the center of the kernel
    # is therefore used.
    half_height_weights, half_width_weights = \
        np.int(np.floor((height_weights-1)/2)), np.int(np.floor((width_weights-1)/2))
    
    # Padding.
    # (pad,) or int is a shortcut for before = after = pad width for all axes. 
    # For instance, pad width = 1 and constant pad value = 0 on this data axis 111 gives 01110 .
    input_layer_padded = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant", constant_values=0)

    for y in range(height_output_layer):
        for x in range(width_output_layer):

            """
            output_layer[:, :, y, x] = \
            np.sum(input_layer_padded[:,:,y*stride:y*stride+height_weights,\
                         x*stride:x*stride+width_weights]\
                           *weight, axis=(1, 2, 3))
            """
            # This was necessary if batch_size > 1 .
            # Otherwise, the broadcasting messes up.
            for channel in range(channels_output_layer):
                output_layer[:, channel, y, x] = \
                np.sum(input_layer_padded[:,:,y*stride:y*stride+height_weights,\
                             x*stride:x*stride+width_weights]\
                               *weight[channel, :, :, :], axis=(1, 2, 3))
                
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return output_layer + (bias)[None, :, None, None]


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    input_layer_gradient, weight_gradient, bias_gradient = None, None, None

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
