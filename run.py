import numpy as np
import matplotlib.pyplot as plt
from factors_depth_reconstruction import FactorGraphDepthReconstruction


if __name__ == '__main__':

    image_i = 4
    img_left = np.load(file='./data/left-{}.npy'.format(image_i))
    img_right = np.load(file='./data/right-{}.npy'.format(image_i))
    img_disparity = np.load(file='./data/disparity-{}.npy'.format(image_i))

    FactorGraphDepthReconstruction(observation=img_disparity, tau=10, max_disparity=64)