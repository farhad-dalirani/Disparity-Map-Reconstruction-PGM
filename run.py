import numpy as np
import matplotlib.pyplot as plt
from factors_depth_reconstruction import FactorGraphDepthReconstruction


if __name__ == '__main__':

    # hyper parameters
    tau = 10
    lambda_value = 1.0
    max_iteration = 12

    for image_i in range(5):    
        print("Image: {}/{}".format(image_i+1, 5))

        # read left and right image and the disoarity map from file
        img_left = np.load(file='./data/left-{}.npy'.format(image_i))
        img_right = np.load(file='./data/right-{}.npy'.format(image_i))
        img_disparity = np.load(file='./data/disparity-{}.npy'.format(image_i))

        # construct Factor Graph for refining disparity map
        factor_graph = FactorGraphDepthReconstruction(observation=img_disparity, tau=tau, max_disparity=64)
        
        # obtain MAP estimation for variables in the Factor Graph
        enhanced_disparity_map = factor_graph.loopy_belief_propagation_maximum_a_posteriori(max_iteration=max_iteration, lambda_value=lambda_value, verbose=True)
        
        plt.figure(figsize=(10, 9))
        # Create the first subplot for left image
        plt.subplot(3, 1, 1)
        plt.imshow(img_left)
        plt.title('Left Image (KITTI Dataset')
        plt.axis('off')
        # Create the first subplot for disparity map
        plt.subplot(3, 1, 2)
        plt.imshow(img_disparity)
        plt.title('Disparity Map (Siamese CNN)')
        plt.axis('off')  
        # Create the second subplot for refined disparity map
        plt.subplot(3, 1, 3)
        plt.imshow(enhanced_disparity_map)
        plt.title('Enhanced Disparity Map By Designed Factor Graph')
        plt.axis('off')  
        plt.tight_layout()
        plt.savefig('./readme_img/{}.png'.format(image_i))
    
