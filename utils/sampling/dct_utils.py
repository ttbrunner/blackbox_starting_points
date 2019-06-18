# MODIFIED: Replaced Perlin Noise with IDCT noise
# See sampling_provider for more information.

# Implemented after Guo et al.: "Low Frequency Adversarial Perturbation"
# https://arxiv.org/abs/1809.08758

# Creates random low-frequency noise with variable frequency.

import numpy as np
from scipy.fftpack import idct


def create_idct_noise(size_px=299, ratio=1./8., normalize=True, rng_gen=None):
    if rng_gen is not None:
        x_freq = rng_gen.standard_normal(size=(1, 3, size_px, size_px), dtype='float32')
    else:
        x_freq = np.random.normal(size=(1, 3, size_px, size_px))

    ratio = float(ratio)

    x = block_idct(x_freq, block_size=size_px, masked=True, ratio=ratio)

    x = np.rollaxis(x, 1, 4)[0, ...]        # Back to channels last,
    if normalize:
        x /= np.linalg.norm(x)

    return x


def block_idct(x, block_size=8, masked=False, ratio=0.5):
    z = np.zeros(shape=x.shape, dtype=np.float32)
    num_blocks = int(x.shape[2] / block_size)
    mask = np.zeros((x.shape[0], x.shape[1], block_size, block_size))
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)]
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho')
    return z
