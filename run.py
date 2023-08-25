import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    image_i = 4
    img_left = np.load(file='./data/left-{}.npy'.format(image_i))
    img_right = np.load(file='./data/right-{}.npy'.format(image_i))
    img_disparity = np.load(file='./data/disparity-{}.npy'.format(image_i))

    plt.figure()
    plt.imshow(img_left)
    plt.figure()
    plt.imshow(img_right)
    plt.figure()
    plt.imshow(img_disparity)
    plt.show()
