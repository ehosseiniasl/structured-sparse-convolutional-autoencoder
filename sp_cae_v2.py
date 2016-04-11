#!/usr/bin/python
"""
2015-11-10 Ehsan Hosseini-Asl

Sparse CAE

Dataset as shared variable

pylearn2 used to load datasets

"""

__author__ = 'ehsanh'

import argparse
import numpy as np
import os
import Queue
import threading
from PIL import Image
import pickle
import cPickle
import random
import sys
import time
import urllib
import theano
import theano.tensor as T
from theano.tensor import nnet
from theano.tensor.signal import downsample
import ipdb
from itertools import izip
import shutil
#import caffe
import cv2
# from sklearn.manifold import TSNE
# theano.config.compute_test_value = 'warn'
import math
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from dlt_utils import tile_raster_images
from pylearn2.datasets.new_norb import NORB
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.binarized_mnist import BinarizedMNIST
import zca_whitening
from theano.tensor.extra_ops import repeat

FLOAT_PRECISION = np.float32
ACT_TANH = 't'
ACT_SIGMOID = 's'
ACT_ReLu = 'r'
ACT_SoftPlus = 'p'

def adadelta_updates(parameters, gradients, rho, eps):

    # create variables to store intermediate updates
    # ipdb.set_trace()
    gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION),) for p in parameters ]
    deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION)) for p in parameters ]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
    # ipdb.set_trace()
    return gradient_sq_updates + deltas_sq_updates + parameters_updates
    # return parameters_updates

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, share_with=None, activation=None):

        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)



        # self.W = W
        # self.b = b

            self.W_delta = theano.shared(
                    np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    borrow=True
                )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

        lin_output = T.dot(self.input, self.W) + self.b

        # ipdb.set_trace()
        if activation == 'tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = nnet.sigmoid(lin_output)
        elif activation == 'relu':
            self.output = T.maximum(lin_output, 0)
        else:
            self.output = lin_output

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class softmaxLayer(object):
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W_delta = theano.shared(
                np.zeros((n_in,n_out), dtype=theano.config.floatX),
                borrow=True
            )

        self.b_delta = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class ConvolutionLayer(object):
    # ACT_TANH = 't'
    # ACT_SIGMOID = 's'
    # ACT_ReLu = 'r'
    # ACT_SoftPlus = 'p'

    def __init__(self, rng, input, filter_shape, poolsize=(2,2), stride=None, if_pool=False, act=None, share_with=None,
                 tied=None, border_mode='valid'):
        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta

        elif tied:
            self.W = tied.W.dimshuffle(1,0,2,3)
            self.b = tied.b

            self.W_delta = tied.W_delta.dimshuffle(1,0,2,3)
            self.b_delta = tied.b_delta

        else:
            fan_in = np.prod(filter_shape[1:])
            poolsize_size = np.prod(poolsize) if poolsize else 1
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / poolsize_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            self.W_delta = theano.shared(
                np.zeros(filter_shape, dtype=theano.config.floatX),
                borrow=True
            )

            # b_update_values = np.zeros((5,filter_shape[0]), dtype=theano.config.floatX)
            self.b_delta = theano.shared(value=b_values, borrow=True)

            #EHA: define update history for momentum gradient
            # self.W_update = theano.shared(
            #     np.zeros(filter_shape, dtype=theano.config.floatX),
            #     borrow=True
            # )
            #
            # # b_update_values = np.zeros((5,filter_shape[0]), dtype=theano.config.floatX)
            # self.b_update = theano.shared(value=b_values, borrow=True)

        #ipdb.set_trace()
        conv_out = nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            border_mode=border_mode)

        #if poolsize:
        if if_pool:
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                st=stride,
                ignore_border=True)
            tmp = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            tmp = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # if act == ConvolutionLayer.ACT_TANH:
        #     self.output = T.tanh(tmp)
        # elif act == ConvolutionLayer.ACT_SIGMOID:
        #     self.output = nnet.sigmoid(tmp)
        # elif act == ConvolutionLayer.ACT_ReLu:
        #     self.output = tmp * (tmp>0)
        # elif act == ConvolutionLayer.ACT_SoftPlus:
        #     self.output = T.log2(1+T.exp(tmp))
        # else:
        #     self.output = tmp
        if act == 'tanh':
            self.output = T.tanh(tmp)
        elif act == 'sigmoid':
            self.output = nnet.sigmoid(tmp)
        elif act == 'relu':
            # self.output = tmp * (tmp>0)
            # self.output = nnet.relu(tmp)
            self.output = 0.5 * (tmp + abs(tmp)) + 1e-9
        elif act == 'softplus':
            # self.output = T.log2(1+T.exp(tmp))
            self.output = nnet.softplus(tmp)
        elif act == 'linear':
            self.output = tmp

        # store parameters of this layer
        self.params = [self.W, self.b]

        #EHA: parameter update- list of 5 previous updates
        # self.params_update = [5*[self.W_update], 5*[self.b_update]]

        self.deltas = [self.W_delta, self.b_delta]

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class CAE(object):
    def __init__(self, data, image_shape, filter_shape, poolsize, sparse_coeff, activation='sigmoid',
                 tied_weight=False, is_linear=False, do_max_pool=False):
        rng = np.random.RandomState(None)
        self.data = data
        self.batchsize = image_shape[0]
        self.in_channels   = image_shape[1]
        self.in_height     = image_shape[2]
        self.in_width      = image_shape[3]
        self.flt_channels  = filter_shape[0]
        self.flt_height    = filter_shape[2]
        self.flt_width     = filter_shape[3]
        self.input = T.ftensor4('input')
        # self.input = input.reshape(image_shape)
        hidden_layer=ConvolutionLayer(rng,
                                      input=self.input,
                                      filter_shape=filter_shape,
                                      act=activation,
                                      border_mode='full',
                                      if_pool=do_max_pool)

        self.hidden_image_shape = (self.batchsize,
                                   self.flt_channels,
                                   self.in_height+self.flt_height-1,
                                   self.in_width+self.flt_width-1)

        self.hidden_pooled_image_shape = (self.batchsize,
                                          self.flt_channels,
                                          (self.in_height+self.flt_height-1)/2,
                                          (self.in_width+self.flt_width-1)/2)

        self.hidden_filter_shape = (self.in_channels,
                                    self.flt_channels,
                                    self.flt_height,
                                    self.flt_width)
        if sparse_coeff == 0:
            if do_max_pool:
                hidden_layer_output = repeat(hidden_layer.output,
                                             repeats=2,
                                             axis=2)
                hidden_layer_output = repeat(hidden_layer_output,
                                             repeats=2,
                                             axis=3)
            else:
                hidden_layer_output = hidden_layer.output
        else:
            feature_map = hidden_layer.output

            # first per featuremap, then across featuremap
            # feature_map_vec = feature_map.reshape((feature_map.shape[0],
            #                                        feature_map.shape[1], feature_map.shape[2]*feature_map.shape[3]))
            # feat_sparsity = feature_map_vec.norm(2, axis=2)
            # feat_sparsity = feat_sparsity.dimshuffle(0, 1, 'x', 'x')
            # feature_map1 = np.divide(feature_map, feat_sparsity+1e-9)
            # examp_sparsity = feature_map1.norm(2, axis=1)
            # examp_sparsity = examp_sparsity.dimshuffle(0, 'x', 1, 2)
            # feature_map2 = np.divide(feature_map1, examp_sparsity+1e-9)

            # first across featuremap, then per featuremap
            examp_sparsity = feature_map.norm(2, axis=1)
            examp_sparsity = examp_sparsity.dimshuffle(0, 'x', 1, 2)
            feature_map1 = np.divide(feature_map, examp_sparsity+1e-9)
            feature_map1_vec = feature_map1.reshape((feature_map1.shape[0],
                                                   feature_map1.shape[1], feature_map1.shape[2]*feature_map1.shape[3]))
            feat_sparsity = feature_map1_vec.norm(2, axis=2)
            feat_sparsity = feat_sparsity.dimshuffle(0, 1, 'x', 'x')
            feature_map2 = np.divide(feature_map1, feat_sparsity+1e-9)

            if do_max_pool:
                hidden_layer_output = repeat(feature_map2,
                                             repeats=2,
                                             axis=2)
                hidden_layer_output = repeat(hidden_layer_output,
                                             repeats=2,
                                             axis=3)
            else:
                hidden_layer_output = feature_map2

        # recon_layer_input = hidden_layer_output

        if is_linear:
            recon_layer=ConvolutionLayer(rng,
                                         input=hidden_layer_output,
                                         filter_shape=self.hidden_filter_shape,
                                         act='linear',
                                         border_mode='valid')
        else:
            recon_layer=ConvolutionLayer(rng,
                                         input=hidden_layer_output,
                                         filter_shape=self.hidden_filter_shape,
                                         act=activation,
                                         border_mode='valid')


        self.tied_weight = tied_weight
        if self.tied_weight:
            # recon_layer.W = hidden_layer.W
            # recon_layer.W = recon_layer.W.dimshuffle(1,0,2,3)
            weight = hidden_layer.W.get_value()
            recon_layer.W.set_value(weight.transpose(1,0,2,3), borrow=True)

        self.layers = [hidden_layer, recon_layer]
        self.params = sum([layer.params for layer in self.layers], [])

        # self.params = hidden_layer.params + recon_layer.params


        L1_sparsity = hidden_layer_output.norm(1, axis=(2, 3))
        # L1_sparsity = T.sum(np.abs(feature_map2), axis=(2, 3))

        # sparse_filter = T.mean(L1_sparsity.sum(axis=1), axis=(0))
        sparse_filter = T.mean(L1_sparsity, axis=(0, 1))

        # sparsity = T.mean(feature_map2, axis=(2,3))
        # sparse_filter = T.mean(sparsity, axis=(0, 1))

        # L=T.sum(T.pow(T.sub(recon_layer.output, self.input), 2), axis=0)
        L=T.sum(T.pow(T.sub(recon_layer.output, self.input), 2), axis=(1,2,3)) # sum over channel,height, width

        cost = 0.5*T.mean(L) + sparse_coeff * sparse_filter

        grads = T.grad(cost, self.params)

        # learning_rate = 0.1
        # updates = [(param_i, param_i-learning_rate*grad_i)
        #            for param_i, grad_i in zip(self.params, grads)]

        updates = adadelta_updates(self.params, grads, rho=0.95, eps=1e-6)

        # self.train = theano.function(
        # [self.input],
        # cost,
        # updates=updates,
        # name="train cae model")
        index = T.lscalar('index')
        batch_begin = index * self.batchsize
        batch_end = batch_begin + self.batchsize

        self.train = theano.function(
                    inputs=[index],
                    outputs=cost,
                    updates=updates,
                    givens={
                        self.input: self.data[batch_begin:batch_end]
                    },
                    name="train cae model")

        self.activation = downsample.max_pool_2d(
                input=hidden_layer.output,
                ds=poolsize,
                ignore_border=True)

        # self.get_activation = theano.function(
        #     [self.input],
        #     self.activation,
        #     updates=None,
        #     name='get hidden activation')
        # num = T.bscalar
        self.get_activation = theano.function(
            inputs=[index],
            # outputs=self.activation,
            outputs=hidden_layer.output if do_max_pool else self.activation,
            updates=None,
            givens={
                self.input: self.data[batch_begin:batch_end]
            },
            name='get hidden activation')

        # self.get_reconstruction = theano.function(
        #                     inputs=[self.input],
        #                     outputs=recon_layer.output,
        #                     updates=None,
        #                     name='get reconstruction')
        self.get_reconstruction = theano.function(
                            inputs=[index],
                            outputs=recon_layer.output,
                            updates=None,
                            givens={
                                self.input: self.data[batch_begin:batch_end]
                            },
                            name='get reconstruction')

    def visualize_filters(self, result_dir, batch):
        print 'save results in '+result_dir
        for i in xrange(self.batchsize):
            feature_map = self.get_activation(batch)
            reconstructed = self.get_reconstruction(batch)

        I = np.zeros((self.batchsize*self.flt_channels, np.prod(self.hidden_pooled_image_shape[-2:])))
        for i in xrange(self.batchsize):
            for j in xrange(self.flt_channels):
                I[i*self.flt_channels+j, :] = feature_map[i, j, :, :].flatten(1)
        image = Image.fromarray(
                                tile_raster_images(X=I,
                                                   img_shape=feature_map.shape[-2:],
                                                   tile_shape=(self.batchsize, self.flt_channels),
                                                   tile_spacing=(2, 2))
                                )
        image.save(result_dir+'featuremap.png')

        I = np.zeros((self.batchsize, reconstructed.shape[2] ** 2))
        for i in xrange(self.batchsize):
            I[i, :] = reconstructed[i,:,:,:].flatten(1)
        if self.batchsize>=8:
            image = Image.fromarray(
                                    tile_raster_images(X=I,
                                                       img_shape=reconstructed.shape[-2:],
                                                       tile_shape=(8, self.batchsize/8),
                                                       tile_spacing=(2, 2))
                                    )
        else:
            image = Image.fromarray(
                                    tile_raster_images(X=I,
                                                       img_shape=reconstructed.shape[-2:],
                                                       tile_shape=(1, self.batchsize),
                                                       tile_spacing=(2, 2))
                                    )
        image.save(result_dir+'reconstructed.png')

        # sample_data = data[:n_images]
        sample_data = self.data.get_value(borrow=True)[batch*self.batchsize:(batch+1)*self.batchsize]
        for i in xrange(self.batchsize):
            I[i, :] = sample_data[i, :, :, :].flatten(1)
        if self.batchsize>=8:
            image = Image.fromarray(
                                    tile_raster_images(X=I,
                                                       img_shape=sample_data.shape[-2:],
                                                       tile_shape=(8, self.batchsize/8),
                                                       tile_spacing=(2, 2))
                                    )
        else:
            image = Image.fromarray(
                                tile_raster_images(X=I,
                                                   img_shape=sample_data.shape[-2:],
                                                   tile_shape=(1, self.batchsize),
                                                   tile_spacing=(2, 2))
                                )
        image.save(result_dir+'data.png')

        I = np.zeros((self.flt_channels, self.flt_height * self.flt_width))
        A = self.layers[0].W.get_value(borrow=True)
        for i in xrange(self.flt_channels):
            I[i, :] = A[i, :, :].flatten()
        image = Image.fromarray(
                                tile_raster_images(X=I,
                                img_shape=(self.flt_height, self.flt_width),
                                tile_shape=(8, self.flt_channels/8),
                                tile_spacing=(2, 2))
                                )
        image.save(result_dir+'filter_image.png')

        I = np.zeros((self.flt_channels, self.flt_height * self.flt_width))
        A = self.layers[1].W.get_value(borrow=True)
        for i in xrange(self.flt_channels):
            I[i, :] = A[0, i, :, :].flatten(1)
        image = Image.fromarray(
                                tile_raster_images(X=I,
                                img_shape=(self.flt_height, self.flt_width),
                                tile_shape=(8, self.flt_channels/8),
                                tile_spacing=(2, 2))
                                )
        image.save(result_dir+'decode_image.png')

    def visualize_colored_filters(self, result_dir, batch):
        print 'save results in '+result_dir
        # n_images = self.batchsize
        for i in xrange(self.batchsize):
            feature_map = self.get_activation(batch)
            reconstructed = self.get_reconstruction(batch)

        I = np.zeros((self.batchsize*self.flt_channels, np.prod(self.hidden_pooled_image_shape[-2:])))
        for i in xrange(self.batchsize):
            for j in xrange(self.flt_channels):
                I[i*self.flt_channels+j, :] = feature_map[i, j, :, :].flatten(1)
        image = Image.fromarray(
                                tile_raster_images(X=I,
                                                   img_shape=feature_map.shape[-2:],
                                                   tile_shape=(self.batchsize, self.flt_channels),
                                                   tile_spacing=(2, 2))
                                )
        image.save(result_dir+'featuremap.png')

        I_R = np.zeros((self.batchsize, np.prod(reconstructed.shape[2:])))
        I_G = np.zeros((self.batchsize, np.prod(reconstructed.shape[2:])))
        I_B = np.zeros((self.batchsize, np.prod(reconstructed.shape[2:])))
        for i in xrange(self.batchsize):
            I_R[i, :] = reconstructed[i, 0, :, :].flatten(1)
            I_G[i, :] = reconstructed[i, 1, :, :].flatten(1)
            I_B[i, :] = reconstructed[i, 2, :, :].flatten(1)
        I = (I_R, I_G, I_B, None)
        if self.batchsize>=8:
            image = Image.fromarray(
                                    tile_raster_images(X=I,
                                                       img_shape=reconstructed.shape[-2:],
                                                       tile_shape=(8, self.batchsize/8),
                                                       tile_spacing=(2, 2))
                                    )
        else:
            image = Image.fromarray(
                                    tile_raster_images(X=I,
                                                       img_shape=reconstructed.shape[-2:],
                                                       tile_shape=(1, self.batchsize),
                                                       tile_spacing=(2, 2))
                                    )
        image.save(result_dir+'reconstructed.png')

        # sample_data = data[:n_images]
        sample_data = self.data.get_value(borrow=True)[batch*self.batchsize:(batch+1)*self.batchsize]
        I_R = np.zeros((self.batchsize, np.prod(sample_data.shape[2:])))
        I_G = np.zeros((self.batchsize, np.prod(sample_data.shape[2:])))
        I_B = np.zeros((self.batchsize, np.prod(sample_data.shape[2:])))
        for i in xrange(self.batchsize):
            I_R[i, :] = sample_data[i, 0, :, :].flatten(1)
            I_G[i, :] = sample_data[i, 1, :, :].flatten(1)
            I_B[i, :] = sample_data[i, 2, :, :].flatten(1)
        I = (I_R, I_G, I_B, None)
        if self.batchsize>=8:
            image = Image.fromarray(
                                    tile_raster_images(X=I,
                                                       img_shape=sample_data.shape[-2:],
                                                       tile_shape=(8, self.batchsize/8),
                                                       tile_spacing=(2, 2))
                                    )
        else:
            image = Image.fromarray(
                                tile_raster_images(X=I,
                                                   img_shape=sample_data.shape[-2:],
                                                   tile_shape=(1, self.batchsize),
                                                   tile_spacing=(2, 2))
                                )
        image.save(result_dir+'data.png')

        I_R = np.zeros((self.flt_channels, self.flt_height**2))
        I_G = np.zeros((self.flt_channels, self.flt_height**2))
        I_B = np.zeros((self.flt_channels, self.flt_height**2))
        A = self.layers[0].W.get_value(borrow=True)
        for i in xrange(self.flt_channels):
            I_R[i, :] = A[i, 0, :, :].flatten()
            I_G[i, :] = A[i, 1, :, :].flatten()
            I_B[i, :] = A[i, 2, :, :].flatten()
        I = (I_R, I_G, I_B, None)
        encode_image = Image.fromarray(
                                tile_raster_images(X=I,
                                img_shape=(self.flt_height, self.flt_width),
                                tile_shape=(8, self.flt_channels/8),
                                tile_spacing=(2, 2))
                                )
        encode_image.save(result_dir+'filter_image.png')

        if not self.tied_weight:
            I_R = np.zeros((self.flt_channels, self.flt_height**2))
            I_G = np.zeros((self.flt_channels, self.flt_height**2))
            I_B = np.zeros((self.flt_channels, self.flt_height**2))
            A = self.layers[1].W.get_value(borrow=True)
            for i in xrange(self.flt_channels):
                I_R[i, :] = A[0, i, :, :].flatten()
                I_G[i, :] = A[1, i, :, :].flatten()
                I_B[i, :] = A[2, i, :, :].flatten()
            I = (I_R, I_G, I_B, None)
            decode_image = Image.fromarray(
                                    tile_raster_images(X=I,
                                    img_shape=(self.flt_height, self.flt_width),
                                    tile_shape=(8, self.flt_channels/8),
                                    tile_spacing=(2, 2))
                                    )
            decode_image.save(result_dir+'decode_image.png')

    def save(self, filename):
        f = open(filename, 'w')
        if self.tied_weight:
            for layer in self.layers[:-1]:
                pickle.dump(layer.get_state(), f, -1)
        else:
            for layer in self.layers:
                pickle.dump(layer.get_state(), f, -1)
        f.close()

    def load(self, filename):
        f = open(filename)
        if self.tied_weight:
            for layer in self.layers[:-1]:
                layer.set_state(pickle.load(f))
        else:
            for layer in self.layers:
                layer.set_state(pickle.load(f))
        f.close()
        print 'model loaded from', filename


class SCAE(CAE):
    def __init__(self, n_layers, image_shape, filter_shape, poolsize, sparsity=0.0, tied_weights=False):
        cae_s = []
        for layer in n_layers:
            if layer == 0:
                input_shape = image_shape
                filter_shape
            cae = CAE()



def do_pretraining_cae(data_name, model, save_name, image_shape, result_dir, max_epoch=1):
    batch_size, c, h, w = image_shape
    # progress_report = 1
    save_interval = 1800
    # num_subjects = data.shape[0]
    num_subjects = model.data.get_value(borrow=True).shape[0]
    num_batches = num_subjects/batch_size
    last_save = time.time()
    epoch = 0
    print 'training CAE'
    sys.stdout.flush()
    # file_head = 'cae_smallNORB_'
    # while True:
    try:
        loss_hist = np.empty((max_epoch,), dtype=np.float32)
        for epoch in xrange(max_epoch):
            # loss = 0
            start_time = time.time()
            #ipdb.set_trace()
            cost_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            for batch in xrange(num_batches):
                # batch_data = data[batch*batch_size:(batch+1)*batch_size]
                # cost_hist[batch] = model.train(batch_data)
                cost_hist[batch] = model.train(batch)
            epoch_time = time.time()-start_time
            loss = np.mean(cost_hist)
            loss_hist[epoch] = loss
            print 'epoch:%d\tcost:%4f\ttime:%2f' % (epoch, loss, epoch_time/60.)
            # print 'epoch:', epoch, ' cost:', loss, ' time:', epoch_time/60., 'min'
            sys.stdout.flush()
            # if epoch % progress_report == 0:
                # loss /= progress_report
                # print '%d\t%g\t%f' % (epoch, loss, time.time()-start_time)
                # sys.stdout.flush()
                # loss = 0
            if time.time() - last_save >= save_interval:
                # loss_history.append(loss)
                # filename = save_name+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                filename=save_name+'.pkl'
                model.save(filename)
                print 'model saved to', filename
                shutil.copy(filename, result_dir)
                shutil.copy('out.log', result_dir)
                do_visualize(data_name=data_name,model=model,result_dir=result_dir)
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                # filename = save_name+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                filename=save_name+'.pkl'
                model.save(filename)
                print 'max epoch reached. model saved to', filename
                shutil.copy(filename, result_dir)
                shutil.copy('out.log', result_dir)
                do_visualize(data_name=data_name,model=model,result_dir=result_dir)
                sys.stdout.flush()
                return filename
            sys.stdout.flush()
    except KeyboardInterrupt:
        # filename = save_name+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
        filename=save_name+'.pkl'
        model.save(filename)
        print 'model saved to', filename
        shutil.copyfile(filename, result_dir)
        shutil.copy('out.log', result_dir)
        do_visualize(data_name=data_name,model=model,result_dir=result_dir)
        sys.stdout.flush()
        return filename

def get_results(model, nkerns, filter_size, results_dir):
    I_R = np.zeros((nkerns, filter_size**2))
    I_G = np.zeros((nkerns, filter_size**2))
    I_B = np.zeros((nkerns, filter_size**2))
    A = model.hidden_layer.W.get_value(borrow=True)
    for i in xrange(nkerns):
        I_R[i, :] = A[i, 0, :, :].flatten()
        I_G[i, :] = A[i, 1, :, :].flatten()
        I_B[i, :] = A[i, 2, :, :].flatten()
    # ipdb.set_trace()
    I = (I_R, I_G, I_B, None)
    image = Image.fromarray(
                            tile_raster_images(X=I,
                            img_shape=(filter_size, filter_size),
                            tile_shape=(8, nkerns/8),
                            tile_spacing=(2, 2))
                            )
    image.save(results_dir+'1st_CAE_filter_image.png')

def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train sparse convolutional autoencoder')
    parser.add_argument('-data', '--data_name', type=str,
                        help='dataset name')
    parser.add_argument('-m', '--model',
                        help='start with this model')
    parser.add_argument('-p', '--pretrain', action='store_true',
                         help='do pretraining')
    parser.add_argument('-t', '--test', action='store_true',
                        help='do testing')
    parser.add_argument('-fn', '--filter_channel', type=int, default=16,
                        help='filter channel')
    parser.add_argument('-fs', '--filter_size', type=int, default=5,
                        help='filter size')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        help='activation function')
    parser.add_argument('-sp', '--sparse_coeff', type=float, default=0,
                        help='sparsity coefficient')
    parser.add_argument('-td', '--tied', action='store_true',
                        help='tied weights')
    parser.add_argument('-ln', '--linear', action='store_true',
                        help='linear CAE')
    parser.add_argument('-zca', '--do_zca', action='store_true',
                        help='ZCA whitening')
    parser.add_argument('-scale', '--do_scale', action='store_true',
                        help='scale image to [0,1]')
    parser.add_argument('-mp', '--do_maxpool', action='store_true',
                        help='do max pooling')
    parser.add_argument('-ep', '--epoch', type=int, default=100,
                        help='epoch number')
    parser.add_argument('-batch', '--batchsize', type=int, default=100,
                        help='batch size')
    args = parser.parse_args()
    return args.data_name, args.model, args.pretrain, args.test, args.filter_channel, \
           args.filter_size, args.activation, args.sparse_coeff, args.tied, \
           args.linear, args.do_zca, args.do_scale, args.do_maxpool, args.epoch, args.batchsize

def do_visualize(data_name, model, result_dir, batch_index=1):
    if data_name in ['cifar', 'svhn', 'cifarw']:
        model.visualize_colored_filters(result_dir=result_dir,
                                        batch=batch_index)
    elif data_name in ['smallNORB', 'mnist', 'bmnist']:
        model.visualize_filters(result_dir=result_dir,
                                batch=batch_index)


def main():
    data_name, model, pretrain, test, filter_channel, \
    filter_size, activation, sparse_coeff, is_tied_weights, \
    is_linear, do_zca, do_scale, do_maxpoool, n_epochs, batch_size = ProcessCommandLine()
    print '... Loading data and parameters'
    print 'dataset: ', data_name
    print 'filter_channel=', filter_channel
    print 'filter_size=', filter_size
    print 'activation=', activation
    print 'sparsity coefficient=', sparse_coeff
    print 'Max pooling=', do_maxpoool
    print 'tied weight=', is_tied_weights
    print 'linear CAE=', is_linear
    print 'ZCA whitening=', do_zca
    print 'Scale image=', do_scale
    print 'Batch size=', batch_size
    print 'number of epoch=', n_epochs
    # batch_size = 100                      # number of images in each batch
    n_epochs = 100                        # number of experiment epochs
    learning_rate = 0.1                   # learning rate of SGD
    # filter_channel = 16                           # number of feature maps in ConvAE
    # dataset = 'data/mnist.pkl.gz'         # address of data
    # rng = np.random.RandomState(23455)  # random generator
    # filter_size = 11
    # n_images = 20
    # sparse_coeff = 1
    if sparse_coeff == 0:
        model_type = 'cae'
    else:
        model_type = 'spcae'

    model_save_name = model_type+'_'+data_name+'_'+ \
                      '[fn=%d,fs=%d,sp=%.3f,maxpool=%s,tied=%s,act=%s,linear=%s,ZCA=%s,scale=%s]' \
                      % (filter_channel, filter_size, sparse_coeff, do_maxpoool, is_tied_weights,
                         activation, is_linear, do_zca, do_scale)
    results_dir = model_save_name+'/'
    # results_dir = data_name+'_'+model_name+'/'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if data_name == 'smallNORB':
        ### load smallNORB train set ###
        # norb = SmallNORB('train', True)
        # data_x = norb.adjust_for_viewer(norb.X)
        # data_y = norb.y
        # pickle.dump((data_x, data_y),open('smallNORB.pkl','wb'), -1)
        # results_dir = 'smallNORB_scae/'
        # results_dir = 'smallNORB_'+model_name+'/'
        # if not os.path.isdir(results_dir):
        #     os.mkdir(results_dir)
        # f = open('smallNORB.pkl', 'r')
        # data, data_y = pickle.load(f)
        # _, feat = data.shape
        # f.close()
        # train = NORB(which_norb='small', which_set='train')
        # ipdb.set_trace()
        # window = preprocessing.CentralWindow(window_shape=(64,64))
        # train.apply_preprocessor(preprocessor=window)
        # train.X = train.X.astype('float32')
        # zca = preprocessing.ZCA()
        # train.apply_preprocessor(preprocessor=zca, can_fit=True)
        # _, feat = train.X.shape
        # data_x = train.X[:, :feat/2]
        norb = sio.loadmat('smallNORB_matlab/smallNORB_train_32x32.mat')
        train_x = norb['trainData'].transpose(1, 0)
        if do_zca:
            zca = zca_whitening.ZCA()
            zca.fit(train_x)
            train_x = zca.transform(train_x)
        if do_scale:
            min_max_scaler = MinMaxScaler()
            train_x_T = min_max_scaler.fit_transform(train_x.T)
            train_x = train_x_T.T
        data_x = train_x
        idx = random.shuffle(range(data_x.shape[0]))
        data_x = data_x[idx, :][0,:,:]
        # ipdb.set_trace()
        im_channel = 1
        im_height = 32
        im_width = 32


    elif data_name == 'cifar':
        ### load cifar10 ###
        # results_dir = 'cifar'+model_name+'/'
        # data_x = pickle.load(open('cifar10.pkl', 'r'))
        train = CIFAR10('train', gcn=55.)
        im_channel = 3
        im_height = 32
        im_width = 32
        # min_max_scaler = MinMaxScaler()
        # data_x = min_max_scaler.fit_transform(data_x)
        # data_x = cifar10.X

    # elif data_name == 'cifarw':
    #     ### load cifar10 ###
    #     # results_dir = 'cifar'+model_name+'/'
    #     # data_x = pickle.load(open('cifar10_whitened.pkl', 'r'))
    #     train = CIFAR10('train', gcn=55.)
    #     # zca = preprocessing.ZCA()
    #     # cifar10.apply_preprocessor(preprocessor=zca, can_fit=True)
    #     im_channel = 3
    #     im_height = 32
    #     im_width = 32
        # min_max_scaler = MinMaxScaler()
        # data_x = min_max_scaler.fit_transform(cifar10.X)
        # data_x = cifar10.X
        # ipdb.set_trace()

    elif data_name == 'svhn':
        f = open('svhn.pkl', 'r')
        svhn = pickle.load(f)
        f.close()
        im_channel = 3
        im_height = 32
        im_width = 32
        train_x, train_y = svhn[0]
        test_x, test_y = svhn[1]
        extra_x, extra_y = svhn[2]
        train_x = train_x.transpose([3, 2, 0, 1])
        test_x = test_x.transpose([3, 2, 0, 1])
        extra_x = extra_x.transpose([3, 2, 0, 1])
        # Scale to [0,1]
        channel_scale_factor = train_x.max(axis=(2, 3)).astype('float32')
        train_x_scaled = train_x/channel_scale_factor.reshape(channel_scale_factor.shape[0], im_channel, 1, 1)
        data_x = train_x_scaled.reshape(train_x.shape[0], im_channel*im_height*im_width)

    elif data_name == 'mnist':
        # f = open('mnist.pkl', 'r')
        # mnist = pickle.load(f)
        # f.close()
        # train_x, train_y = mnist[0]
        # valid_x, valid_y = mnist[1]
        # test_x, test_y = mnist[2]
        # data_x = train_x
        train = MNIST('train')
        # zca = preprocessing.ZCA()
        # train.apply_preprocessor(preprocessor=zca, can_fit=True)
        # ipdb.set_trace()
        # min_max_scaler = MinMaxScaler()
        # data_x = min_max_scaler.fit_transform(train.X)
        # data_x = train.X
        # data_y = train.y
        im_channel = 1
        im_height = 28
        im_width = 28

    elif data_name == 'bmnist':
        train = BinarizedMNIST(which_set='train')
        # zca = preprocessing.ZCA()
        # train.apply_preprocessor(preprocessor=zca, can_fit=True)
        # ipdb.set_trace()
        # min_max_scaler = MinMaxScaler()
        # data_x = min_max_scaler.fit_transform(train.X)
        # data_x = train.X
        # data_y = train.y
        im_channel = 1
        im_height = 28
        im_width = 28

    if do_zca and data_name not in ['smallNORB', 'svhn']:
        zca = preprocessing.ZCA()
        train.apply_preprocessor(preprocessor=zca, can_fit=True)
        data_x = train.X
        pass
    if do_scale and data_name not in ['smallNORB', 'svhn']:
        min_max_scaler = MinMaxScaler()
        data_x_T = min_max_scaler.fit_transform(train.X.T)
        data_x = data_x_T.T
        pass
    if not do_zca and not do_scale and data_name not in ['smallNORB', 'svhn']:
        data_x = train.X
    # if data_name not in ['smallNORB']:
    #     data_x = train.X

    n_samples, n_feat = data_x.shape
    data_x = data_x.reshape((n_samples, im_channel, im_height, im_width))
    if data_name == 'mnist':
        data_x = data_x.transpose(0,1,3,2)
    train_set_x = theano.shared(np.asarray(data_x, dtype=np.float32), borrow=True)

    # image_shp = (batch_size, im_channel, data_x.shape[2], data_x.shape[3])
    image_shp = (batch_size, im_channel, im_height, im_width)
    filter_shp = (filter_channel, im_channel, filter_size, filter_size)

    print 'building model'
    cae1 = CAE(image_shape=image_shp,
               data=train_set_x,
               filter_shape=filter_shp,
               poolsize=(2, 2),
               sparse_coeff=sparse_coeff,
               activation=activation,
               do_max_pool=do_maxpoool,
               tied_weight=is_tied_weights,
               is_linear=is_linear)
    print 'model built'
    sys.stdout.flush()

    if model:
        cae1.load(model)
        pass

    if pretrain:
        do_pretraining_cae(data_name=data_name,
                           model=cae1,
                           save_name=model_save_name,
                           image_shape=image_shp,
                           result_dir=results_dir,
                           max_epoch=n_epochs)
    elif test:
        do_visualize(data_name=data_name,
                     model=cae1,
                     result_dir=results_dir)


if __name__ == '__main__':
    sys.exit(main())
