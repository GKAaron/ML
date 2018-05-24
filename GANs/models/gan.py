"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])
        self.learning_rate_placeholder = tf.placeholder(tf.float32,[])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        self.dis,self.gen = self.update_op(self.d_loss,self.g_loss,
                                self.learning_rate_placeholder)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            nn = layers.fully_connected(x,100)
            y = layers.fully_connected(nn,1,activation_fn=None)
            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        y_t = tf.ones(shape=tf.shape(y))
        y_f = tf.zeros(shape=tf.shape(y_hat))
        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_t,logits=y))+\
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_f,logits=y_hat))
        return -l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            nn = layers.fully_connected(z,100)
            x_hat = layers.fully_connected(nn,self._ndims)
            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        y_f = tf.zeros(shape=tf.shape(y_hat))
        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             labels=y_f,logits=y_hat))
        return l

    def update_op(self,d_loss,g_loss,learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        dis = optimizer.minimize(d_loss,var_list=tf.get_collection(key=
                                 tf.GraphKeys.TRAINABLE_VARIABLES,scope=
                                 'discriminator'))
        gen = optimizer.minimize(g_loss,var_list=tf.get_collection(key=
                                 tf.GraphKeys.TRAINABLE_VARIABLES,scope=
                                 'generator'))
        return dis,gen

    def generate_samples(self,z_np):
        return self.session.run(self.x_hat,feed_dict={self.z_placeholder:z_np})
