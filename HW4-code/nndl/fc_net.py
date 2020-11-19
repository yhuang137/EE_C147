# import numpy as np
# import pdb

# from .layers import *
# from .layer_utils import *

# """
# This code was originally written for CS 231n at Stanford University
# (cs231n.stanford.edu).  It has been modified in various areas for use in the
# ECE 239AS class at UCLA.  This includes the descriptions of what code to
# implement as well as some slight potential changes in variable names to be
# consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
# permission to use this code.  To see the original version, please visit
# cs231n.stanford.edu.
# """


# class TwoLayerNet(object):
#     """
#     A two-layer fully-connected neural network with ReLU nonlinearity and
#     softmax loss that uses a modular layer design. We assume an input dimension
#     of D, a hidden dimension of H, and perform classification over C classes.

#     The architecure should be affine - relu - affine - softmax.

#     Note that this class does not implement gradient descent; instead, it
#     will interact with a separate Solver object that is responsible for running
#     optimization.

#     The learnable parameters of the model are stored in the dictionary
#     self.params that maps parameter names to numpy arrays.
#     """

#     def __init__(
#         self,
#         input_dim=3 * 32 * 32,
#         hidden_dims=100,
#         num_classes=10,
#         dropout=0,
#         weight_scale=1e-3,
#         reg=0.0,
#     ):
#         """
#         Initialize a new network.

#         Inputs:
#         - input_dim: An integer giving the size of the input
#         - hidden_dims: An integer giving the size of the hidden layer
#         - num_classes: An integer giving the number of classes to classify
#         - dropout: Scalar between 0 and 1 giving dropout strength.
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - reg: Scalar giving L2 regularization strength.
#         """
#         self.params = {}
#         self.reg = reg

#         # ================================================================ #
#         # YOUR CODE HERE:
#         #   Initialize W1, W2, b1, and b2.  Store these as self.params['W1'],
#         #   self.params['W2'], self.params['b1'] and self.params['b2']. The
#         #   biases are initialized to zero and the weights are initialized
#         #   so that each parameter has mean 0 and standard deviation weight_scale.
#         #   The dimensions of W1 should be (input_dim, hidden_dim) and the
#         #   dimensions of W2 should be (hidden_dims, num_classes)
#         # ================================================================ #
#         self.params["b1"] = np.zeros((hidden_dims, 1))
#         self.params["b2"] = np.zeros((num_classes, 1))
#         self.params["W1"] = np.random.randn(input_dim, hidden_dims) * weight_scale
#         self.params["W2"] = np.random.randn(hidden_dims, num_classes) * weight_scale
#         # ================================================================ #
#         # END YOUR CODE HERE
#         # ================================================================ #

#     def loss(self, X, y=None):
#         """
#         Compute loss and gradient for a minibatch of data.

#         Inputs:
#         - X: Array of input data of shape (N, d_1, ..., d_k)
#         - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

#         Returns:
#         If y is None, then run a test-time forward pass of the model and return:
#         - scores: Array of shape (N, C) giving classification scores, where
#           scores[i, c] is the classification score for X[i] and class c.

#         If y is not None, then run a training-time forward and backward pass and
#         return a tuple of:
#         - loss: Scalar value giving the loss
#         - grads: Dictionary with the same keys as self.params, mapping parameter
#           names to gradients of the loss with respect to those parameters.
#         """
#         scores = None

#         # ================================================================ #
#         # YOUR CODE HERE:
#         #   Implement the forward pass of the two-layer neural network. Store
#         #   the class scores as the variable 'scores'.  Be sure to use the layers
#         #   you prior implemented.
#         # ================================================================ #
#         h1, cache1 = affine_forward(X, self.params["W1"], self.params["b1"])
#         r2, cache2 = relu_forward(h1)
#         scores, cache3 = affine_forward(r2, self.params["W2"], self.params["b2"])
#         # ================================================================ #
#         # END YOUR CODE HERE
#         # ================================================================ #

#         # If y is None then we are in test mode so just return scores
#         if y is None:
#             return scores

#         loss, grads = 0, {}
#         # ================================================================ #
#         # YOUR CODE HERE:
#         #   Implement the backward pass of the two-layer neural net.  Store
#         #   the loss as the variable 'loss' and store the gradients in the
#         #   'grads' dictionary.  For the grads dictionary, grads['W1'] holds
#         #   the gradient for W1, grads['b1'] holds the gradient for b1, etc.
#         #   i.e., grads[k] holds the gradient for self.params[k].
#         #
#         #   Add L2 regularization, where there is an added cost 0.5*self.reg*W^2
#         #   for each W.  Be sure to include the 0.5 multiplying factor to
#         #   match our implementation.
#         #
#         #   And be sure to use the layers you prior implemented.
#         # ================================================================ #
#         loss_soft, dx = softmax_loss(scores, y)
#         loss = loss_soft + 0.5 * self.reg * (
#             np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
#         )
#         dh1, dW2, db2 = affine_backward(dx, cache3)
#         grads["W2"] = dW2 + self.reg * self.params["W2"]
#         grads["b2"] = db2.T

#         dr = relu_backward(dh1, cache2)
#         dx2, dW1, db1 = affine_backward(dr, cache1)
#         grads["W1"] = dW1 + self.reg * self.params["W1"]
#         grads["b1"] = db1.T
#         # ================================================================ #
#         # END YOUR CODE HERE
#         # ================================================================ #

#         return loss, grads


# class FullyConnectedNet(object):
#     """
#   A fully-connected neural network with an arbitrary number of hidden layers,
#   ReLU nonlinearities, and a softmax loss function. This will also implement
#   dropout and batch normalization as options. For a network with L layers,
#   the architecture will be

#   {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

#   where batch normalization and dropout are optional, and the {...} block is
#   repeated L - 1 times.

#   Similar to the TwoLayerNet above, learnable parameters are stored in the
#   self.params dictionary and will be learned using the Solver class.
#   """

#     def __init__(
#         self,
#         hidden_dims,
#         input_dim=3 * 32 * 32,
#         num_classes=10,
#         dropout=0,
#         use_batchnorm=False,
#         reg=0.0,
#         weight_scale=1e-2,
#         dtype=np.float32,
#         seed=None,
#     ):
#         """
#     Initialize a new FullyConnectedNet.

#     Inputs:
#     - hidden_dims: A list of integers giving the size of each hidden layer.
#     - input_dim: An integer giving the size of the input.
#     - num_classes: An integer giving the number of classes to classify.
#     - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
#       the network should not use dropout at all.
#     - use_batchnorm: Whether or not the network should use batch normalization.
#     - reg: Scalar giving L2 regularization strength.
#     - weight_scale: Scalar giving the standard deviation for random
#       initialization of the weights.
#     - dtype: A numpy datatype object; all computations will be performed using
#       this datatype. float32 is faster but less accurate, so you should use
#       float64 for numeric gradient checking.
#     - seed: If not None, then pass this random seed to the dropout layers. This
#       will make the dropout layers deteriminstic so we can gradient check the
#       model.
#     """
#         self.use_batchnorm = use_batchnorm
#         self.use_dropout = dropout > 0
#         self.reg = reg
#         self.num_layers = 1 + len(hidden_dims)
#         self.dtype = dtype
#         self.params = {}

#         # ================================================================ #
#         # YOUR CODE HERE:
#         #   Initialize all parameters of the network in the self.params dictionary.
#         #   The weights and biases of layer 1 are W1 and b1; and in general the
#         #   weights and biases of layer i are Wi and bi. The
#         #   biases are initialized to zero and the weights are initialized
#         #   so that each parameter has mean 0 and standard deviation weight_scale.
#         #
#         #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
#         #   parameters to zero.  The gamma and beta parameters for layer 1 should
#         #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
#         #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm
#         #   is true and DO NOT batch normalize the output scores.
#         # ================================================================ #

#         x_dim = input_dim
#         for idx, hidden_dim in enumerate(hidden_dims):
#             self.params["W" + str(idx + 1)] = (
#                 np.random.randn(x_dim, hidden_dim) * weight_scale
#             )

#             self.params["b" + str(idx + 1)] = np.zeros(hidden_dim)
#             x_dim = hidden_dim

#             if self.use_batchnorm == True:
#                 self.params["gamma" + str(idx + 1)] = np.ones(hidden_dim)
#                 self.params["beta" + str(idx + 1)] = np.zeros(hidden_dim)

#         # output layer
#         self.params["W" + str(self.num_layers)] = (
#             np.random.randn(x_dim, num_classes) * weight_scale
#         )
#         self.params["b" + str(self.num_layers)] = np.zeros(num_classes)

#         # ================================================================ #
#         # END YOUR CODE HERE
#         # ================================================================ #

#         # When using dropout we need to pass a dropout_param dictionary to each
#         # dropout layer so that the layer knows the dropout probability and the mode
#         # (train / test). You can pass the same dropout_param to each dropout layer.
#         self.dropout_param = {}
#         if self.use_dropout:
#             self.dropout_param = {"mode": "train", "p": dropout}
#             if seed is not None:
#                 self.dropout_param["seed"] = seed

#         # With batch normalization we need to keep track of running means and
#         # variances, so we need to pass a special bn_param object to each batch
#         # normalization layer. You should pass self.bn_params[0] to the forward pass
#         # of the first batch normalization layer, self.bn_params[1] to the forward
#         # pass of the second batch normalization layer, etc.
#         self.bn_params = []
#         if self.use_batchnorm:
#             self.bn_params = [{"mode": "train"} for i in np.arange(self.num_layers - 1)]

#         # Cast all parameters to the correct datatype
#         for k, v in self.params.items():
#             self.params[k] = v.astype(dtype)

#     def loss(self, X, y=None):
#         """
#     Compute loss and gradient for the fully-connected net.
#     Input / output: Same as TwoLayerNet above.
#     """
#         X = X.astype(self.dtype)
#         mode = "test" if y is None else "train"

#         # Set train/test mode for batchnorm params and dropout param since they
#         # behave differently during training and testing.
#         if self.dropout_param is not None:
#             self.dropout_param["mode"] = mode
#         if self.use_batchnorm:
#             for bn_param in self.bn_params:
#                 bn_param[mode] = mode

#         scores = None

#         # ================================================================ #
#         # YOUR CODE HERE:
#         #   Implement the forward pass of the FC net and store the output
#         #   scores as the variable "scores".
#         #
#         #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
#         #   between the affine_forward and relu_forward layers.  You may
#         #   also write an affine_batchnorm_relu() function in layer_utils.py.
#         #
#         #   DROPOUT: If dropout is non-zero, insert a dropout layer after
#         #   every ReLU layer.
#         # ================================================================ #
#         affine_cache, relu_cache, batchnorm_cache, dropout_cache = {}, {}, {}, {}
#         X = np.reshape(X, [X.shape[0], -1])
#         for i in range(len(hidden_dims)):
#             affine_out, affine_cache[str(i + 1)] = affine_forward(
#                 X, self.params["W" + str(i + 1)], self.params["b" + str(i + 1)]
#             )

#             if self.use_batchnorm == True
#                 batchnorm_out, batchnorm_cache[str(i + 1)] = batchnorm_forward(
#                     affine_out,
#                     self.params["gamma" + str(i + 1)],
#                     self.params["beta" + str(i + 1)],
#                     self.bn_params[i],
#                 )
#                 relu_input = batchnorm_out
#             else:
#                 relu_input = affine_out

#             relu_out, relu_cache[str(i + 1)] = relu_forward(relu_input)

#             if self.use_dropout == True
#                 relu_out, dropout_cache[str(i + 1)] = dropout_forward(
#                     relu_out, self.dropout_param
#                 )

#         scores, out_cache = affine_forward(
#             relu_out,
#             self.params["W" + str(self.num_layers)],
#             self.params["b" + str(self.num_layers)],
#         )

#         # ================================================================ #
#         # END YOUR CODE HERE
#         # ================================================================ #

#         # If test mode return early
#         if mode == "test":
#             return scores

#         loss, grads = 0.0, {}
#         # ================================================================ #
#         # YOUR CODE HERE:
#         #   Implement the backwards pass of the FC net and store the gradients
#         #   in the grads dict, so that grads[k] is the gradient of self.params[k]
#         #   Be sure your L2 regularization includes a 0.5 factor.
#         #
#         #   BATCHNORM: Incorporate the backward pass of the batchnorm.
#         #
#         #   DROPOUT: Incorporate the backward pass of dropout.
#         # ================================================================ #
#         softmax_loss, dx = softmax_loss(scores, y)
#         loss = softmax_loss + (
#             0.5
#             * self.reg
#             * (np.sum(np.square(self.params["W" + str(self.num_layers)])))
#         )
#         # BATCHNORM: Incorporate the backward pass of the batchnorm.
#         dx_backward, dw_backward, db_backward = affine_backward(dx, out_cache)
#         grads["W" + str(self.num_layers)] = (
#             dw_backward + self.reg * self.params["W" + str(self.num_layers)]
#         )
#         grads["b" + str(self.num_layers)] = db_backward

#         for i in range(len(hidden_dims), 0, -1):
#             if self.use_dropout == True:
#                 dx_back = dropout_backward(dx_back, dropout_cache[str(i)])
#             dx_relu = relu_backward(dx_back, relu_cache[str(i)])
#             if self.use_batchnorm == True:
#                 dx_bnorm, dgamma, dbeta = batchnorm_backward(
#                     dx_relu, batchnorm_cache[str(i)]
#                 )
#                 # update
#                 affine_backward_input = dx_bnorm
#                 grads["gamma" + str(i)] = dgamma
#                 grads["beta" + str(i)] = dbeta

#             dx_backward, dw_backward, db_backward = affine_backward(
#                 dx_relu, affine_cache[str(i)]
#             )

#             grads["W" + str(i)] = dw_backward + self.reg * self.params["W" + str(i)]
#             grads["b" + str(i)] = db_backward
#             loss += 0.5 * self.reg * (np.sum(np.square(self.params["W" + str(i)])))

#         # ================================================================ #
#         # END YOUR CODE HERE
#         # ================================================================ #

#         return loss, grads


import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""


class FullyConnectedNet(object):
    """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=0,
        use_batchnorm=False,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize all parameters of the network in the self.params dictionary.
        #   The weights and biases of layer 1 are W1 and b1; and in general the
        #   weights and biases of layer i are Wi and bi. The
        #   biases are initialized to zero and the weights are initialized
        #   so that each parameter has mean 0 and standard deviation weight_scale.
        #
        #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
        #   parameters to zero.  The gamma and beta parameters for layer 1 should
        #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
        #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm
        #   is true and DO NOT batch normalize the output scores.
        # ================================================================ #

        cur_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            # initialize weights and bias
            self.params["W" + str(idx + 1)] = (
                np.random.randn(cur_dim, hidden_dim) * weight_scale
            )
            self.params["b" + str(idx + 1)] = np.zeros(hidden_dim)

            # initialize gammas and betas
            if self.use_batchnorm:
                self.params["gamma" + str(idx + 1)] = np.ones(hidden_dim)
                self.params["beta" + str(idx + 1)] = np.zeros(hidden_dim)

            cur_dim = hidden_dim

        self.params["W" + str(self.num_layers)] = (
            np.random.randn(cur_dim, num_classes) * weight_scale
        )
        self.params["b" + str(self.num_layers)] = np.zeros(num_classes)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{"mode": "train"} for i in np.arange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
    Compute loss and gradient for the fully-connected net.
    Input / output: Same as TwoLayerNet above.
    """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param["mode"] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the FC net and store the output
        #   scores as the variable "scores".
        #
        #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
        #   between the affine_forward and relu_forward layers.  You may
        #   also write an affine_batchnorm_relu() function in layer_utils.py.
        #
        #   DROPOUT: If dropout is non-zero, insert a dropout layer after
        #   every ReLU layer.
        # ================================================================ #

        # initialize caches
        fc_cache = {}
        relu_cache = {}
        batchnorm_cache = {}
        dropout_cache = {}

        # flatten image
        X = np.reshape(X, [X.shape[0], -1])

        # go through all layers
        for i in range(self.num_layers - 1):
            # fc layer
            fc_out, fc_cache[str(i + 1)] = affine_forward(
                X, self.params["W" + str(i + 1)], self.params["b" + str(i + 1)]
            )

            # batchnorm layer
            relu_input = fc_out
            if self.use_batchnorm:
                batchnorm_out, batchnorm_cache[str(i + 1)] = batchnorm_forward(
                    fc_out,
                    self.params["gamma" + str(i + 1)],
                    self.params["beta" + str(i + 1)],
                    self.bn_params[i],
                )
                relu_input = batchnorm_out
            relu_out, relu_cache[str(i + 1)] = relu_forward(relu_input)

            # dropout layer
            if self.use_dropout:
                relu_out, dropout_cache[str(i + 1)] = dropout_forward(
                    relu_out, self.dropout_param
                )

            # update X
            X = relu_out.copy()

        # output FC layer with no relu
        scores, final_cache = affine_forward(
            X,
            self.params["W" + str(self.num_layers)],
            self.params["b" + str(self.num_layers)],
        )

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backwards pass of the FC net and store the gradients
        #   in the grads dict, so that grads[k] is the gradient of self.params[k]
        #   Be sure your L2 regularization includes a 0.5 factor.
        #
        #   BATCHNORM: Incorporate the backward pass of the batchnorm.
        #
        #   DROPOUT: Incorporate the backward pass of dropout.
        # ================================================================ #

        # initialize
        loss, dx = softmax_loss(scores, y)
        loss += (
            0.5
            * self.reg
            * (np.sum(np.square(self.params["W" + str(self.num_layers)])))
        )
        dx_back, dw_back, db_back = affine_backward(dx, final_cache)
        grads["W" + str(self.num_layers)] = (
            dw_back + self.reg * self.params["W" + str(self.num_layers)]
        )
        grads["b" + str(self.num_layers)] = db_back

        # go backward all layers and update weights, bias, gammas and betas
        for i in range(self.num_layers - 1, 0, -1):
            # dropout layer
            if self.use_dropout:
                dx_back = dropout_backward(dx_back, dropout_cache[str(i)])
            dx_relu = relu_backward(dx_back, relu_cache[str(i)])

            # batchnorm layer
            affine_backward_input = dx_relu
            if self.use_batchnorm:
                dx_bn, dgamma, dbeta = batchnorm_backward(
                    dx_relu, batchnorm_cache[str(i)]
                )
                grads["gamma" + str(i)] = dgamma
                grads["beta" + str(i)] = dbeta
                affine_backward_input = dx_bn
            dx_back, dw_back, db_back = affine_backward(
                affine_backward_input, fc_cache[str(i)]
            )

            grads["W" + str(i)] = dw_back + self.reg * self.params["W" + str(i)]
            grads["b" + str(i)] = db_back
            loss += 0.5 * self.reg * (np.sum(np.square(self.params["W" + str(i)])))

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads

