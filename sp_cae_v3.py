#!/usr/bin/python
"""
2015-11-16 Ehsan Hosseini-Asl

Sparse CAE

Dataset as shared variable

pylearn2 used to load datasets

Stacked layer of SP_CAE Added

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
                (n_in, n_out),
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
            outputs=self.activation,
            # outputs=hidden_layer.output if do_max_pool else self.activation,
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
        self.data = data
        self.batchsize = image_shape[0]
        self.in_channels   = image_shape[1]
        self.in_height     = image_shape[2]
        self.in_width      = image_shape[3]
        self.flt_channels  = filter_shape[0]
        self.flt_height    = filter_shape[2]
        self.flt_width     = filter_shape[3]
        self.input = T.ftensor4('input')
        cae_s = []
        for layer in n_layers:
            if layer == 0:
                input_shape = image_shape
                filter_shape
            cae = CAE()

class stacked_CAE(object):
    def __init__(self, data_x, data_y, test_x, test_y, image_shape, filter_shapes, poolsize,
                 sparse_coeff, activation=None, num_classes=2):
        rng = np.random.RandomState(None)
        self.images = T.ftensor4(name='images')
        self.labels = T.ivector('labels')

        self.data_x = data_x
        self.data_y = data_y
        self.test_x = test_x
        self.test_y = test_y
        self.batchsize = image_shape[0]
        self.in_channels   = image_shape[1]
        self.in_height     = image_shape[2]
        self.in_width      = image_shape[3]
        self.filter_shapes = filter_shapes
        # self.flt_channels  = filter_shapes[0]
        # self.flt_height    = filter_shapes[2]
        # self.flt_width     = filter_shapes[3]
        self.flt_channels1  = filter_shapes[0][0]
        self.flt_channels2  = filter_shapes[1][0]
        self.flt_channels3  = filter_shapes[2][0]
        self.flt_height    = filter_shapes[0][2]
        self.flt_width     = filter_shapes[0][3]

        conv1=ConvolutionLayer(rng,
                               input=self.images,
                               filter_shape=filter_shapes[0],
                               act=activation,
                               if_pool=True,
                               poolsize=poolsize,
                               border_mode='valid')

        self.conv1_output_shape = (self.batchsize,
                              self.flt_channels1,
                              (self.in_height-self.flt_height+1)/2,
                              (self.in_width-self.flt_width+1)/2)

        # l2-normalization
        if sparse_coeff!=0:
            feature_map = conv1.output
            examp_sparsity = feature_map.norm(2, axis=1)
            examp_sparsity = examp_sparsity.dimshuffle(0, 'x', 1, 2)
            feature_map1 = np.divide(feature_map, examp_sparsity+1e-9)
            feature_map1_vec = feature_map1.reshape((feature_map1.shape[0],
                                                   feature_map1.shape[1], feature_map1.shape[2]*feature_map1.shape[3]))
            feat_sparsity = feature_map1_vec.norm(2, axis=2)
            feat_sparsity = feat_sparsity.dimshuffle(0, 1, 'x', 'x')
            conv1_output = np.divide(feature_map1, feat_sparsity+1e-9)
        else:
            conv1_output = conv1.output

        conv2 = ConvolutionLayer(rng,
                                 input=conv1_output,
                                 filter_shape=filter_shapes[1],
                                 act=activation,
                                 if_pool=True,
                                 poolsize=poolsize,
                                 border_mode='valid')

        self.conv2_output_shape = (self.batchsize,
                              self.flt_channels2,
                              (self.conv1_output_shape[2]-self.flt_height+1)/2,
                              (self.conv1_output_shape[3]-self.flt_width+1)/2)

        # l2-normalization
        if sparse_coeff!=0:
            feature_map = conv2.output
            examp_sparsity = feature_map.norm(2, axis=1)
            examp_sparsity = examp_sparsity.dimshuffle(0, 'x', 1, 2)
            feature_map1 = np.divide(feature_map, examp_sparsity+1e-9)
            feature_map1_vec = feature_map1.reshape((feature_map1.shape[0],
                                                   feature_map1.shape[1], feature_map1.shape[2]*feature_map1.shape[3]))
            feat_sparsity = feature_map1_vec.norm(2, axis=2)
            feat_sparsity = feat_sparsity.dimshuffle(0, 1, 'x', 'x')
            conv2_output = np.divide(feature_map1, feat_sparsity+1e-9)
        else:
            conv2_output = conv2.output

        conv3 = ConvolutionLayer(rng,
                                 input=conv2_output,
                                 filter_shape=filter_shapes[2],
                                 act=activation,
                                 if_pool=True,
                                 poolsize=poolsize,
                                 border_mode='valid')

        # l2-normalization
        if sparse_coeff!=0:
            feature_map = conv3.output
            examp_sparsity = feature_map.norm(2, axis=1)
            examp_sparsity = examp_sparsity.dimshuffle(0, 'x', 1, 2)
            feature_map1 = np.divide(feature_map, examp_sparsity+1e-9)
            feature_map1_vec = feature_map1.reshape((feature_map1.shape[0],
                                                   feature_map1.shape[1], feature_map1.shape[2]*feature_map1.shape[3]))
            feat_sparsity = feature_map1_vec.norm(2, axis=2)
            feat_sparsity = feat_sparsity.dimshuffle(0, 1, 'x', 'x')
            conv3_output = np.divide(feature_map1, feat_sparsity+1e-9)
        else:
            conv3_output = conv3.output

        self.conv3_output_shape = (self.batchsize,
                              self.flt_channels3,
                              (self.conv2_output_shape[2]-self.flt_height+1)/2,
                              (self.conv2_output_shape[3]-self.flt_width+1)/2)

        # for layer in hidden_size:
        # ip1_input=conv3.output.flatten(2)
        # ip1 = HiddenLayer(input=ip1_input,
        #                    n_in=conv3_output_shape,
        #                    n_out=hidden_size[0],
        #                    activation='sigmoid')
        #
        # ip2 = HiddenLayer(input=ip1.output,
        #                    n_in=hidden_size[0],
        #                    n_out=hidden_size[1],
        #                    activation='sigmoid')
        #
        # ip3 = HiddenLayer(input=ip2.output,
        #                    n_in=hidden_size[1],
        #                    n_out=hidden_size[2],
        #                    activation='sigmoid')
        #
        # ip4 = HiddenLayer(input=ip3.output,
        #                    n_in=hidden_size[2],
        #                    n_out=hidden_size[3],
        #                    activation='sigmoid')

        # output_layer_input = conv3.output.flatten(2)
        output_layer_input = conv3_output.flatten(2)
        output_layer = softmaxLayer(input=output_layer_input,
                                    n_in=np.prod(self.conv3_output_shape[1:]),
                                    n_out=num_classes)

        L1_sparsity = (conv1_output.norm(1, axis=(2, 3))+
                       conv2_output.norm(1, axis=(2, 3))+
                       conv3_output.norm(1, axis=(2, 3)))/3.
        sparse_filter = T.mean(L1_sparsity, axis=(0, 1))

        self.layers = [conv1,
                       conv2,
                       conv3,
                       output_layer]

        self.params = sum([l.params for l in self.layers], [])
        # self.cost = output_layer.negative_log_likelihood(self.labels)
        self.cost = output_layer.negative_log_likelihood(self.labels) + sparse_coeff * sparse_filter
        self.grads = T.grad(self.cost, self.params)

        self.updates = adadelta_updates(parameters=self.params,
                                        gradients=self.grads,
                                        rho=0.95,
                                        eps=1e-6)

        self.error = output_layer.errors(self.labels)

        index = T.lscalar('index')
        batch_begin = index * self.batchsize
        batch_end = batch_begin + self.batchsize

        self.train = theano.function(
                    inputs=[index],
                    outputs=[self.cost, self.error],
                    updates=self.updates,
                    givens={
                        self.images: self.data_x[batch_begin:batch_end],
                        self.labels: self.data_y[batch_begin:batch_end]
                    },
                    name="train scae model")

        self.test = theano.function(
                    inputs=[index],
                    outputs=[self.cost, self.error],
                    givens={
                        self.images: self.data_x[batch_begin:batch_end],
                        self.labels: self.data_y[batch_begin:batch_end]
                    },
                    name="test scae model")

    def visualize_filters(self, result_dir):
        print 'save results in '+result_dir
        for idx, layer in enumerate(self.layers[:-1]):
            A = layer.W.get_value(borrow=True)
            # if A.shape[1] == 1:
            #     I = np.zeros((self.filter_shapes[idx][0], np.prod(self.filter_shapes[idx][1:])))
            #     for i in xrange(self.filter_shapes[idx][0]):
            #         I[i, :] = A[i, :, :].flatten()
            #     image = Image.fromarray(
            #         tile_raster_images(X=I,
            #                            img_shape=(self.filter_shapes[idx][2], self.filter_shapes[idx][3]),
            #                            tile_shape=(8, np.prod(self.filter_shapes[idx][:2])/8),
            #                            tile_spacing=(2, 2))
            #     )
            #     image_name = 'layer%d_filter_image.png' % (idx)
            #     image.save(result_dir+image_name)
            # else:
            I = np.zeros((np.prod(self.filter_shapes[idx][:2]), np.prod(self.filter_shapes[idx][2:])))
            for f in xrange(A.shape[0]):
                for i in range(A.shape[1]):
                    I[f*A.shape[1]+i, :] = A[f, i, :, :].flatten()
            image = Image.fromarray(
                tile_raster_images(X=I,
                                   img_shape=(self.filter_shapes[idx][2], self.filter_shapes[idx][3]),
                                   tile_shape=(8, np.prod(self.filter_shapes[idx][:2])/8),
                                   tile_spacing=(2, 2))
            )
            image_name = 'layer%d_filter_image.png' % (idx)
            image.save(result_dir+image_name)



    def visualize_colored_filters(self, result_dir):
        print 'save results in '+result_dir
        for idx, layer in enumerate(self.layers[:-1]):
            I_R = np.zeros((self.filter_shapes[idx][0], self.filter_shapes[idx][2] * self.filter_shapes[idx][3]))
            I_G = np.zeros((self.filter_shapes[idx][0], self.filter_shapes[idx][2] * self.filter_shapes[idx][3]))
            I_B = np.zeros((self.filter_shapes[idx][0], self.filter_shapes[idx][2] * self.filter_shapes[idx][3]))
            A = layer.W.get_value(borrow=True)
            for i in xrange(self.filter_shapes[idx][0]):
                I_R[i, :] = A[i, 0, :, :].flatten()
                I_G[i, :] = A[i, 1, :, :].flatten()
                I_B[i, :] = A[i, 2, :, :].flatten()
            I = (I_R, I_G, I_B, None)
            encode_image = Image.fromarray(
                                    tile_raster_images(X=I,
                                    img_shape=(self.filter_shapes[idx][2], self.filter_shapes[idx][3]),
                                    tile_shape=(8, self.filter_shapes[idx][0]/8),
                                    tile_spacing=(2, 2))
                                    )
            encode_image.save(result_dir+'layer'+str(layer)+'filter_image.png')

    def load_cae(self, filename, cae_layer):
        f = open(filename)
        self.layers[cae_layer].set_state(pickle.load(f))
        print 'scae convolutional layer %d loaded from %s' % (cae_layer, filename)

    def save(self, filename):
        f = open(filename, 'w')
        for l in self.layers:
            pickle.dump(l.get_state(), f, -1)
        f.close()
        print 'scae model saved to', filename

    def load(self, filename):
        f = open(filename)
        for l in self.layers:
            l.set_state(pickle.load(f))
        f.close()
        print 'scae model loaded from', filename


def do_pretraining_cae(data_name, models, save_name, result_dir, cae_layer, max_epoch=1):
    batch_size, c, h, w = models[0].batchsize, models[0].flt_channels, models[0].flt_height, models[0].flt_width
    save_interval = 1800
    num_subjects = models[0].data.get_value(borrow=True).shape[0]
    num_batches = num_subjects/batch_size
    last_save = time.time()
    print 'training CAE_'+str(cae_layer)
    sys.stdout.flush()
    try:
        loss_hist = np.empty((max_epoch,), dtype=np.float32)
        for epoch in xrange(max_epoch):
            start_time = time.time()
            cost_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            for batch in xrange(num_batches):
                cost_hist[batch] = models[cae_layer-1].train(batch)
            epoch_time = time.time()-start_time
            loss = np.mean(cost_hist)
            loss_hist[epoch] = loss
            print 'epoch:%d\tcost:%4f\ttime:%2f' % (epoch, loss, epoch_time/60.)
            sys.stdout.flush()
            if time.time() - last_save >= save_interval:
                filename=save_name+'.pkl'
                models[cae_layer-1].save(filename)
                # print 'model saved to', filename
                do_visualize(data_name=data_name, model=models[cae_layer-1], result_dir=result_dir, layer=cae_layer)
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                filename=save_name+'.pkl'
                models[cae_layer-1].save(filename)
                print 'max epoch reached. model saved to', filename
                shutil.copy(filename, result_dir)
                if cae_layer==1:
                    do_visualize(data_name=data_name, model=models[cae_layer-1], result_dir=result_dir)
                sys.stdout.flush()
                return filename
            sys.stdout.flush()
    except KeyboardInterrupt:
        filename=save_name+'.pkl'
        models[cae_layer-1].save(filename)
        # print 'model saved to', filename
        shutil.copyfile(filename, result_dir)
        do_visualize(data_name=data_name, model=models[cae_layer-1], result_dir=result_dir)
        sys.stdout.flush()
        return filename

def do_testing(model):
    num_subjects = model.test_x.get_value(borrow=True).shape[0]
    num_batches = num_subjects/model.batchsize
    print 'testing stacked CAE'
    cost_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
    error_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
    for batch in xrange(num_batches):
        cost, error = model.test(batch)
        cost_hist[batch] = cost
        error_hist[batch] = error
    loss = np.mean(cost_hist)
    total_error = np.mean(error_hist)
    return loss, total_error


def do_finetuning(data_name, model, save_name, result_dir, max_epoch=1):
    save_interval = 1800
    num_subjects = model.data_x.get_value(borrow=True).shape[0]
    num_batches = num_subjects/model.batchsize
    last_save = time.time()
    print 'finetuning stacked CAE'
    sys.stdout.flush()
    print 'epoch\tcost\terror\ttime'
    try:
        loss_hist = np.empty((max_epoch,), dtype=np.float32)
        epochs_error = np.empty((max_epoch,), dtype=FLOAT_PRECISION)
        epochs_loss = np.empty((max_epoch,), dtype=FLOAT_PRECISION)
        for epoch in xrange(max_epoch):
            start_time = time.time()
            cost_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            error_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            for batch in xrange(num_batches):
                cost, error = model.train(batch)
                cost_hist[batch] = cost
                error_hist[batch] = error
            epoch_time = time.time()-start_time
            loss = np.mean(cost_hist)
            epoch_error = np.mean(error_hist)
            epochs_error[epoch] = epoch_error
            epochs_loss[epoch] = loss
            loss_hist[epoch] = loss
            print '%d\t%.4f\t%.4f\t%.2f' % (epoch, loss, epoch_error, epoch_time/60.)
            sys.stdout.flush()
            if time.time() - last_save >= save_interval:
                filename = save_name+'.pkl'
                model.save(filename)
                # print 'scae model saved to', filename
                test_loss, test_error = do_testing(model)
                print 'testing scae\n'
                print 'error:%f\tloss:%f\n' % (test_loss, test_error)
                # stacked_model_visualize(data_name=data_name, model=model, result_dir=result_dir)
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                filename = save_name+'.pkl'
                model.save(filename)
                print 'max epoch reached. model saved to', filename
                shutil.copy(filename, result_dir)
                test_loss, test_error = do_testing(model)
                print '\ntesting scae'
                print 'error:%f\tloss:%f\n' % (test_error, test_loss)
                f = open(save_name+'_train_log', 'w')
                pickle.dump((epochs_loss, epochs_error), f, -1)
                f.close()
                print 'training log saved'
                # stacked_model_visualize(data_name=data_name, model=model, result_dir=result_dir)
                sys.stdout.flush()
                return filename
            sys.stdout.flush()
    except KeyboardInterrupt:
        filename=save_name+'.pkl'
        model.save(filename)
        # print 'model saved to', filename
        shutil.copyfile(filename, result_dir)
        do_visualize(data_name=data_name, model=model, result_dir=result_dir)
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
    # parser.add_argument('-m', '--model',
    #                     help='start with this model')
    parser.add_argument('-cae1', '--cae1_model',
                        help='Initialize cae1 model')
    parser.add_argument('-cae2', '--cae2_model',
                        help='Initialize cae2 model')
    parser.add_argument('-cae3', '--cae3_model',
                        help='Initialize cae3 model')
    parser.add_argument('-p', '--pretrain', type=int, default=0,
                        help='pretrain cae layer')
    parser.add_argument('-dp', '--do_pretrain', action='store_true',
                        help='do cae pretraining')
    parser.add_argument('-t', '--do_test', action='store_true',
                        help='do testing')
    parser.add_argument('-ft', '--finetune', action='store_true',
                        help='do fine tuning')
    parser.add_argument('-dv', '--visualize', action='store_true',
                        help='do visualize')
    parser.add_argument('-fn', '--filter_channel', type=int, default=[8,8,8], nargs='+',
                        help='filter channel list')
    parser.add_argument('-fs', '--filter_size', type=int, default=[3,3,3], nargs='+',
                        help='filter size list')
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
    parser.add_argument('-ep', '--epoch', type=int, default=100,
                        help='epoch number')
    parser.add_argument('-batch', '--batchsize', type=int, default=100,
                        help='batch size')
    parser.add_argument('-model', '--model_name', type=str, default=None,
                        help='load model name')
    args = parser.parse_args()
    return args.data_name, args.cae1_model, args.cae2_model, args.cae3_model, args.pretrain, \
           args.do_pretrain, args.do_test, \
           args.finetune, args.visualize, \
           args.filter_channel, args.filter_size, args.activation, args.sparse_coeff, args.tied, \
           args.linear, args.do_zca, args.do_scale, args.epoch, args.batchsize, args.model_name

def do_visualize(data_name, model, result_dir, batch_index=1, layer=1):
    if data_name in ['cifar', 'svhn', 'cifarw']:
        model.visualize_colored_filters(result_dir=result_dir,
                                        batch=batch_index)
    elif data_name in ['smallNORB', 'mnist', 'bmnist']:
        model.visualize_filters(result_dir=result_dir,
                                batch=batch_index)

def stacked_model_visualize(data_name, model, result_dir):
    if data_name in ['cifar', 'svhn', 'cifarw']:
        model.visualize_colored_filters(result_dir=result_dir)
    elif data_name in ['smallNORB', 'mnist', 'bmnist']:
        model.visualize_filters(result_dir=result_dir)

def get_activation(model):
    # batch_size, _, _, _ = model.image_shape
    num_subjects = model.data.get_value(borrow=True).shape[0]
    batch_size, c, h, w = model.hidden_pooled_image_shape
    num_batches = num_subjects/batch_size
    hidden_activation = np.empty((num_subjects, c, h, w), dtype=FLOAT_PRECISION)
    for batch in xrange(num_batches):
        hidden_activation[batch*batch_size:(batch+1)*batch_size] = model.get_activation(batch)
    return hidden_activation



def main():
    data_name, cae1_model, cae2_model, cae3_model, pretrain, do_pretrain, do_test, fine_tune, visualize, \
    filter_channel, \
    filter_size, activation, sparse_coeff, is_tied_weights, \
    is_linear, do_zca, do_scale, n_epochs, batch_size, model_name = ProcessCommandLine()
    print '... Loading data and parameters'
    print 'dataset: ', data_name
    print 'filter_channel=', filter_channel
    print 'filter_size=', filter_size
    print 'activation=', activation
    print 'sparsity coefficient=', sparse_coeff
    print 'tied weight=', is_tied_weights
    print 'linear CAE=', is_linear
    print 'ZCA whitening=', do_zca
    print 'Scale image=', do_scale
    print 'Batch size=', batch_size
    print 'number of epoch=', n_epochs
    if model_name:
        print 'load %s' % model_name

    n_epochs = 100                        # number of experiment epochs
    learning_rate = 0.1                   # learning rate of SGD
    if sparse_coeff == 0:
        model_type = 'cae'
    else:
        model_type = 'spcae'

    if do_pretrain:
        model_save_name = model_type+str(pretrain)+'_'+data_name+'_'+ \
                          '[fn=%d,fs=%d,sp=%.3f,tied=%s,act=%s,linear=%s,ZCA=%s,scale=%s]' \
                          % (filter_channel[pretrain-1], filter_size[pretrain-1],
                             sparse_coeff, is_tied_weights,
                             activation, is_linear, do_zca, do_scale)
        results_dir = model_save_name+'/'
    else:
        model_save_name = 'stacked_'+model_type+'_'+data_name+'_'+ \
                          '[fn1=%d,fn2=%d,fn3=%d,fs1=%d,fs2=%d,fs3=%d,sp=%.3f,tied=%s,act=%s,linear=%s,ZCA=%s,' \
                          'scale=%s]' \
                          % (filter_channel[0], filter_channel[1], filter_channel[2], filter_size[0], filter_size[1],
                             filter_size[2],
                             sparse_coeff, is_tied_weights, activation, is_linear, do_zca, do_scale)
        results_dir = model_save_name+'/'

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if data_name == 'smallNORB':
        norb_train = sio.loadmat('smallNORB_matlab/smallNORB_train_32x32.mat')
        train_x = norb_train['trainData'].transpose(1, 0)
        train_y = norb_train['trainLabels']
        norb_test = sio.loadmat('smallNORB_matlab/smallNORB_test_32x32.mat')
        test_x = norb_test['testData'].transpose(1, 0)
        test_y = norb_test['testLabels']
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
        data_y = train_y
        # ipdb.set_trace()
        im_channel = 1
        im_height = 32
        im_width = 32


    elif data_name == 'cifar':
        train = CIFAR10('train', gcn=55.)
        test = CIFAR10('test', gcn=55.)
        data_y = train.y
        test_x = test.X
        test_y = test.y
        im_channel = 3
        im_height = 32
        im_width = 32

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
        train = MNIST('train')
        test = MNIST('test')
        # test = MNIST('test')
        data_y = train.y
        test_x = test.X
        test_y = test.y
        im_channel = 1
        im_height = 28
        im_width = 28

    elif data_name == 'bmnist':
        train = BinarizedMNIST(which_set='train')
        data_y = train.y
        im_channel = 1
        im_height = 28
        im_width = 28

    if do_zca and data_name not in ['smallNORB', 'svhn']:
        zca = preprocessing.ZCA()
        train.apply_preprocessor(preprocessor=zca, can_fit=True)
        data_x = train.X
        train.apply_preprocessor(preprocessor=zca, can_fit=True)
        test_x = test.X
        pass
    if do_scale and data_name not in ['smallNORB', 'svhn']:
        min_max_scaler = MinMaxScaler()
        data_x_T = min_max_scaler.fit_transform(train.X.T)
        data_x = data_x_T.T
        test_x_T = min_max_scaler.fit_transform(test.X.T)
        test_x = test_x_T.T
        pass
    if not do_zca and not do_scale and data_name not in ['smallNORB', 'svhn']:
        data_x = train.X
        test_x = test.X

    n_samples, n_feat = data_x.shape
    data_x = data_x.reshape((n_samples, im_channel, im_height, im_width))
    train_set_x = theano.shared(np.asarray(data_x, dtype=np.float32), borrow=True)
    # data_y = train.y
    train_set_y = theano.shared(np.asarray(data_y[:,0], dtype=np.int32), borrow=True)
    # train_set_y = theano.shared(np.asarray(data_y, dtype=np.int32), borrow=True)
    # ipdb.set_trace()
    n_samples_test, _ = test_x.shape
    test_x = test_x.reshape((n_samples_test, im_channel, im_height, im_width))
    test_set_x = theano.shared(np.asarray(test_x, dtype=np.float32), borrow=True)
    # test_y = test.y
    test_set_y = theano.shared(np.asarray(test_y[:,0], dtype=np.int32), borrow=True)
    image_shp = (batch_size, im_channel, im_height, im_width)
    filter_shp_1 = (filter_channel[0], im_channel, filter_size[0], filter_size[0])
    filter_shp_2 = (filter_channel[1], filter_channel[0], filter_size[1], filter_size[1])
    filter_shp_3 = (filter_channel[2], filter_channel[1], filter_size[2], filter_size[2])

    # print 'building model'

    if do_pretrain:
        cae1 = CAE(image_shape=image_shp,
               data=train_set_x,
               filter_shape=filter_shp_1,
               poolsize=(2, 2),
               sparse_coeff=sparse_coeff,
               activation=activation,
               tied_weight=is_tied_weights,
               is_linear=is_linear)
        print 'cae1 built'
        if cae1_model:
            cae1.load(cae1_model)
            print 'cae1 model loaded by:', cae1_model
            pass
        sys.stdout.flush()

        cae1_hidden_data = get_activation(cae1)
        cae1_hidden_set_x = theano.shared(np.asarray(cae1_hidden_data, dtype=np.float32), borrow=True)
        cae2 = CAE(image_shape=cae1.hidden_pooled_image_shape,
                   data=cae1_hidden_set_x,
                   filter_shape=filter_shp_2,
                   poolsize=(2, 2),
                   sparse_coeff=sparse_coeff,
                   activation=activation,
                   tied_weight=is_tied_weights,
                   is_linear=is_linear)
        print 'cae2 built'
        if cae2_model:
            cae2.load(cae2_model)
            print 'cae2 model loaded by:', cae2_model
            pass
        sys.stdout.flush()

        cae2_hidden_data = get_activation(cae2)
        cae2_hidden_set_x = theano.shared(np.asarray(cae2_hidden_data, dtype=np.float32), borrow=True)
        cae3 = CAE(image_shape=cae2.hidden_pooled_image_shape,
                   data=cae2_hidden_set_x,
                   filter_shape=filter_shp_3,
                   poolsize=(2, 2),
                   sparse_coeff=sparse_coeff,
                   activation=activation,
                   tied_weight=is_tied_weights,
                   is_linear=is_linear)
        print 'cae3 built'
        if cae3_model:
            cae3.load(cae3_model)
            print 'cae3 model loaded by:', cae3_model
            pass
        sys.stdout.flush()

        cae_models = [cae1, cae2, cae3]

        if pretrain !=0 and not do_test:
            do_pretraining_cae(data_name=data_name,
                               models=cae_models,
                               save_name=model_save_name,
                               result_dir=results_dir,
                               cae_layer=pretrain,
                               max_epoch=n_epochs)
        elif pretrain !=0 and do_test:
            do_visualize(data_name=data_name,
                         model=cae_models[pretrain-1],
                         result_dir=results_dir,
                         layer=pretrain)
    elif fine_tune:
        # ipdb.set_trace()
        stacked_model = stacked_CAE(data_x=train_set_x,
                                    data_y=train_set_y,
                                    test_x=test_set_x,
                                    test_y=test_set_y,
                                    image_shape=image_shp,
                                    filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
                                    poolsize=(2, 2),
                                    sparse_coeff=sparse_coeff,
                                    activation=activation,
                                    num_classes=np.unique(train_set_y.get_value()).shape[0])
        print 'scae built'
        if cae1_model:
            stacked_model.load_cae(cae1_model, cae_layer=0)
            pass
        if cae2_model:
            stacked_model.load_cae(cae2_model, cae_layer=1)
            pass
        if cae3_model:
            stacked_model.load_cae(cae3_model, cae_layer=2)
            pass

        # for layer, _ in enumerate(stacked_model.layers[:-1]):
        #     stacked_model.load_cae(cae1_model, layer)

        do_finetuning(data_name=data_name,
                      model=stacked_model,
                      save_name=model_save_name,
                      result_dir=results_dir,
                      max_epoch=n_epochs)
    elif visualize:
        print 'visualzie %s' % model_name
        stacked_model = stacked_CAE(data_x=train_set_x,
                                    data_y=train_set_y,
                                    test_x=test_set_x,
                                    test_y=test_set_y,
                                    image_shape=image_shp,
                                    filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
                                    poolsize=(2, 2),
                                    sparse_coeff=sparse_coeff,
                                    activation=activation,
                                    num_classes=np.unique(train_set_y.get_value()).shape[0])
        stacked_model.load(filename=model_name)
        if not os.path.isdir(model_name[:-4]):
            os.mkdir(model_name[:-4])
        stacked_model_visualize(data_name=data_name,
                     model=stacked_model,
                     result_dir=model_name[:-4]+'/')

if __name__ == '__main__':
    sys.exit(main())
