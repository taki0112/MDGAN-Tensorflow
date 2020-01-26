
"""
https://github.com/haihabi/MD-GAN/blob/aaf1df4063a6b7e99d137c3879f3054f9a48dadc/examples/training_md_gan_25_gaussions.ipynb

iteration = 10k
lambda_lr = 0.01
bot_dim = 9


### training process ###
lambda_training_data = tf.constant(value=1.0, dtype=tf.float32, shape=[1])
lambda_net_logit = lambda_net(lambda_training_data, bot_dim) # [1, bot_dim]
# lambda_net = FC 1 + scaled_sigmoid

lambda_lk = gaussian_likelihood_sum(lambda_net_logit, bot_dim)
lambda_loss = tf.reduce_mean(-tf.log(1e-8 + lambda_1k))

### after training ###
lambda_net_logit = lambda_net(lambda_training_data, bot_dim) # [1, bot_dim]
lambda_lk = tf.reduce_sum(gaussian_likelihood_sum(lambda_net_logit, bot_dim))

### example ###
if bot_dim = 9 & lambda_lr = 0.01
lambda_1k = 0.02208706922829151
"""

import numpy as np
import tensorflow as tf

def scaled_sigmoid(x, scale=5.0, shift=2.5):

    x = scale * tf.sigmoid(x) - shift

    return x

def simplex_coordinates(m):
    x = np.zeros([m, m + 1])  # Start with a zero matrix
    np.fill_diagonal(x, 1.0)  # fill diagonal with ones

    x[:, m] = (1.0 - np.sqrt(float(1 + m))) / float(m)  # fill the last column

    c = np.sum(x, axis=1) / (m + 1)  # calculate each row mean
    x = x - np.expand_dims(c, axis=1)  # subtract each row mean

    s = 0.0
    for i in range(0, m):
        s = s + x[i, 0] ** 2
        s = np.sqrt(s)

    return x / s


def var2cov(bot_dim, ngmm):
    cov = np.zeros((bot_dim, bot_dim))
    for k_ in range(bot_dim):
        cov[k_, k_] = 1.
    sigma_real_batch = []
    for c in range(ngmm):
        sigma_real_batch.append(cov)
    return np.array(sigma_real_batch, dtype=np.float32).squeeze().astype('float32') * .25


def simplex_params(bot_dim=9, convert_tensor=True):
    ngmm = bot_dim + 1
    mu_real_batch = simplex_coordinates(bot_dim)
    sigma_real = var2cov(bot_dim, ngmm).astype('float32')
    mu_real = np.array(mu_real_batch.T, dtype=np.float32)
    w_real = (np.ones((ngmm,)) / ngmm).astype('float32')
    sigma_det_rsqrt = np.power(np.linalg.det(2 * np.pi * sigma_real), -0.5)
    sigma_inv = np.linalg.inv(sigma_real)

    if convert_tensor:
        mu_real = tf.convert_to_tensor(mu_real, tf.float32) # [10, 9]
        sigma_real = tf.convert_to_tensor(sigma_real, tf.float32) # [10, 9, 9]
        w_real = tf.convert_to_tensor(w_real, tf.float32) # [10, ]
        sigma_det_rsqrt = tf.convert_to_tensor(sigma_det_rsqrt, tf.float32) # [10, ]
        sigma_inv = tf.convert_to_tensor(sigma_inv, tf.float32) # [10, 9, 9]

    return mu_real, sigma_real, w_real, sigma_det_rsqrt, sigma_inv

def gaussian_likelihood_sum(logit, bot_dim=9):
    mu, sigma, w, sigma_det_rsqrt, sigma_inv = simplex_params(bot_dim=bot_dim)

    logit = tf.expand_dims(logit, axis=1) # [bs, 1, 9]
    mu = tf.expand_dims(mu, axis=0) # [1, 10, 9]
    e_center = tf.expand_dims(logit - mu, axis=-1) # [bs, 10, 9, 1]

    x = tf.matmul(e_center, sigma_inv, transpose_b=True)
    x = tf.matmul(x, e_center, transpose_b=True)
    exp_value = tf.exp(-0.5 * x) # [bs, 10, 1, 9]

    sigma_det_rsqrt = tf.reshape(sigma_det_rsqrt, [1, -1, 1, 1])
    w = tf.reshape(w, [1, -1, 1, 1])

    likelihood = tf.reshape(tf.reduce_sum((w * sigma_det_rsqrt * exp_value), axis=1), [-1])

    return likelihood

def generator_loss(fake_logit):
    lambda_1k = 0.02208706922829151
    epsilon = 1e-8
    bot_dim = 9

    loss = tf.log(epsilon + lambda_1k - gaussian_likelihood_sum(fake_logit, bot_dim))
    loss = tf.reduce_mean(loss)

    return loss

def discriminator_loss(real_logit):
    lambda_1k = 0.02208706922829151
    epsilon = 1e-8
    bot_dim = 9

    loss = tf.log(epsilon + lambda_1k - gaussian_likelihood_sum(real_logit, bot_dim))
    loss = tf.reduce_mean(loss)

    return loss