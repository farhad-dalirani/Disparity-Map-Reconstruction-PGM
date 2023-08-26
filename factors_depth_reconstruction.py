import numpy as np

class FactorGraphDepthReconstruction:

    def __init__(self, observation, tau, max_disparity=64):
        
        # initial depth map that comes from an out of shelf method
        self.observation = observation

        # image size
        self.width = observation.shape[1]
        self.height = observation.shape[0]
        
        # number of random variable in the gragh
        self.num_variables = self.width * self.height

        # uses to truncate value for pairwise factor to stio them become so small
        self.tau = tau

        # different values that a pixel can get: {0, 1, 2, ..., max_disparity}
        self.num_state = max_disparity + 1

        self.factors = []
        # create unary factors
        for row_i in range(self.height):
            for col_j in range(self.width):
                self.factors.append({'tpye':'unary', 'variables': [row_i * self.width + col_j]})
        
        # create pair factors
        for row_i in range(self.height):
            for col_j in range(self.width):
                if row_i < self.height - 1:
                    self.factors.append({'tpye':'pair', 'variables': [row_i * self.width + col_j, (row_i+1) * self.width + col_j]})
                if col_j < self.width - 1:
                    self.factors.append({'tpye':'pair', 'variables': [row_i * self.width + col_j, row_i * self.width + (col_j + 1)]})
        
        # messages type: variable to factor
        dic_variable_to_factor = {}
        # messages type: factor to variable
        dic_factor_to_variable = {}
        # variable's neighbours
        dic_variable_neighbours = {variable_idx: [] for variable_idx in range(self.num_variables)}

        for factor_idx, factor_i in enumerate(self.factors):
            for variable_idx in factor_i['variables']:
                # initial message between a variable to a factor for each state value is zero because later
                # log representation is used for optimizing 
                dic_variable_to_factor[(variable_idx, factor_idx)] = np.zeros(shape=(self.num_state, 1))
                dic_factor_to_variable[(factor_idx, variable_idx)] = np.zeros(shape=(self.num_state, 1))
                # add factor to list of variable neighbours
                dic_variable_neighbours[variable_idx].append(factor_idx)