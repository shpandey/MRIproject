#from tensorlayer.prepro import *
import numpy as np
import skimage.measure
import scipy
from time import localtime, strftime
import logging
import tensorflow as tf
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def distort_img(x):
    x = (x + 1.) / 2.
    # x = flip_axis(x, axis=1, is_random=True)
    # x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
    x = rotation(x, rg=10, is_random=True, fill_mode='constant')
    x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    x = zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    x = brightness(x, gamma=0.05, is_random=True)
    x = x * 2 - 1
    return x


def to_bad_img(x, mask):
    x = (x + 1.) / 2.
    fft = scipy.fftpack.fft2(x[:, :, 0])
    fft = scipy.fftpack.fftshift(fft)
    fft = fft * mask
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    x = np.abs(x)
    x = x * 2 - 1
    return x[:, :, np.newaxis]


def fft_abs_for_map_fn(x):
    x = (x + 1.) / 2.
    x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs


def ssim(data):
    x_good, x_bad = data
    x_good = np.squeeze(x_good)
    x_bad = np.squeeze(x_bad)
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def psnr(data):
    x_good, x_bad = data
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


def vgg_prepro(x):
    x = imresize(x, [244, 244], interp='bilinear', mode=None)
    x = np.tile(x, 3)
    x = x / 127.5 - 1
    return x


def logging_setup(log_dir):
    current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))

    log_all = logging.getLogger('log_all')
    log_all.setLevel(logging.DEBUG)
    log_all.addHandler(logging.FileHandler(log_all_filename))

    log_eval = logging.getLogger('log_eval')
    log_eval.setLevel(logging.INFO)
    log_eval.addHandler(logging.FileHandler(log_eval_filename))

    log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))

    log_50 = logging.getLogger('log_50')
    log_50.setLevel(logging.DEBUG)
    log_50.addHandler(logging.FileHandler(log_50_filename))

    return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename


def apply_random_deformation(images):
    assert len(images.shape) == 4
    [batch_length, x, y, z] = images.shape

    for i in range(0, batch_length):
        images[i, :, :, :] = random_elastic_deformation(images[i, :, :, :])

    return images


def random_elastic_deformation(image, alpha=500, sigma=20, mode='nearest',
                               random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = (np.repeat(np.ravel(x + dx), channels),
               np.repeat(np.ravel(y + dy), channels),
               np.tile(np.arange(channels), height * width))

    values = map_coordinates(image, indices, order=1, mode=mode)

    return values.reshape((height, width, channels))


if __name__ == "__main__":
    pass