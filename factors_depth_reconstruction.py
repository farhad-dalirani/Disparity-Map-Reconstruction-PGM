import numpy as np

class FactorGraphDepthReconstruction:
    """
    A class for creating a factor graph to reconstruct stereo disparity images.

    This class encapsulates the creation and management of a factor graph, which is a graphical representation 
    to refine stereo disparity. The factor graph is designed to incorporate both
    unary and pairwise factors, which collectively drive the reconstruction process.

    The unary factor attempts to maintain the value of a pixel as the value in the 
    provided disparity map. As disparity usually changes smoothly on each object in the scene, 
    the pairwise factors strive to keep the disparity changes of neighboring pixels smooth.
    """
    def __init__(self, observation, tau=10, max_disparity=64):
        """Initialize Variables and Factors for the Factor Graph to reconstruct given disparity map.
           Each pixel is represented as a random variable with a discrete distribution, capable of taking integer values
           ranging from 0 to the maximum disparity. Each variable is interconnected through four factors, corresponding
           to their 4-grid neighbors. These factors play a role in maintaining smoothness among adjacent pixels. The design
           of these four factors is informed by our prior knowledge about the scene: it is expected that significant changes
           in disparity occur primarily at the edges of objects, whereas disparities within objects tend to change more gradually.
           In addition to the four smoothness-controlling factors, each variable is linked to a unary factor. The purpose of
           this unary factor is to encourage the random variable's value to align with the corresponding value in the input
           disparity map. By incorporating this factor, the factor graph seeks to maintain consistency between the estimated
           disparities and the provided disparity map.   

           Factor Graph for an image of size 3 * 3:
             
             F   F   F
            /   /   /
            V-F-V-F-V 
            |   |   |
            | F | F | F
            |/  |/  |/
            V-F-V-F-V 
            |   |   |
            | F | F | F
            |/  |/  |/
            V-F-V-F-V 
        
        Args:
            observation: input disparity map.
            tau: this hyperparameter controls the penalty for ensuring that the absolute difference between the disparities of two neighbors does not exceed a threshold.
            max_disparity: Maximum disparity map in the input disparity map.
        """
        # initial disparity map that comes from an out of shelf method
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
                self.factors.append({'type':'unary', 'variables': [row_i * self.width + col_j]})
        
        # create pair factors
        for row_i in range(self.height):
            for col_j in range(self.width):
                if row_i < self.height - 1:
                    self.factors.append({'type':'pair', 'variables': [row_i * self.width + col_j, (row_i+1) * self.width + col_j]})
                if col_j < self.width - 1:
                    self.factors.append({'type':'pair', 'variables': [row_i * self.width + col_j, row_i * self.width + (col_j + 1)]})
        
        # messages type: variable to factor
        self.dic_variable_to_factor_message = {}
        # messages type: factor to variable
        self.dic_factor_to_variable_message = {}
        # factor neighbours of variable
        self.dic_variable_neighbours = {variable_idx: [] for variable_idx in range(self.num_variables)}

        # initial messages
        for factor_idx, factor_i in enumerate(self.factors):
            for variable_idx in factor_i['variables']:
                # initial message between a variable to a factor for each state value is zero because later
                # log representation is used for optimizing 
                self.dic_variable_to_factor_message[(variable_idx, factor_idx)] = np.zeros(shape=(self.num_state, 1))
                self.dic_factor_to_variable_message[(factor_idx, variable_idx)] = np.zeros(shape=(self.num_state, 1))
                # add factor to list of variable neighbours
                self.dic_variable_neighbours[variable_idx].append(factor_idx)

        # value for unary factor of each variable
        self.unary_value = np.ones(shape=(self.num_variables, self.num_state)) * (1.0/np.e)
        for row_i in range(self.height):
            for col_j in range(self.width):
                self.unary_value[row_i * self.width + col_j, self.observation[row_i, col_j]] = 1
        # log value for unary factor of each variable
        self.unary_value_log = np.log(self.unary_value)

        # values for pairwise factors for each possible state pairs
        self.pairwise_value = np.zeros(shape=(self.num_state, self.num_state))
        for state_i in range(self.num_state):
            for state_j in range(self.num_state):
                self.pairwise_value[state_i, state_j] = max(np.exp(-1 * np.abs(state_i-state_j)), np.exp(-1 * self.tau))
        # log values for pairwise factors for each possible state pairs
        self.pairwise_value_log = np.log(self.pairwise_value)

    def loopy_belief_propagation_maximum_a_posteriori(self, max_iteration, lambda_value, verbose=False):
        """Perform Loopy Belief Propagation to estimate the Maximum A Posteriori (MAP) estimation.
           This method iteratively refines the belief estimates of each variable node based on messages
           from neighboring factor nodes. The process continues for a maximum of 'max_iteration' times.
           It uses logarithms on the original formulation to mitigate certain numerical calculation issues.

           For each iteration:
                - Update messages from factors to variable nodes.
                - Update messages from variable nodes to factor nodes.   
           After the iterations:
           - Compute the Max-Marginals.
           - Derive the Maximum A Posteriori (MAP) Solution.


        Args:
            max_iteration: maximum epochs of message passing
            lambda_value: determines effect of smoothness factors. Higher value means more smoothness.

        Returns:
            refind disparity map
        """
        for iter_i in range(max_iteration):
            
            if verbose == True:
                print('Iteration {}/{}.'.format(iter_i+1, max_iteration))

            # for all factor to variable messages
            for dic_key in self.dic_factor_to_variable_message.keys():
                
                # factor source, variable target
                factor_idx, variable_idx = dic_key

                # if factor is unary
                if self.factors[factor_idx]['type'] == 'unary':
                    self.dic_factor_to_variable_message[(factor_idx, variable_idx)] = np.reshape(self.unary_value_log[variable_idx, :], newshape=(-1, 1))

                elif self.factors[factor_idx]['type'] == 'pair':
                    
                    # other variable involved in factor
                    factor_other_variable = None
                    if variable_idx == self.factors[factor_idx]['variables'][0]:
                        factor_other_variable = self.factors[factor_idx]['variables'][1]
                    else:
                        factor_other_variable = self.factors[factor_idx]['variables'][0]

                    # update message from factor to variable
                    self.dic_factor_to_variable_message[(factor_idx, variable_idx)] = np.max(
                                self.pairwise_value_log + self.dic_variable_to_factor_message[(factor_other_variable, factor_idx)], axis=1, keepdims=True)
                else:
                    raise ValueError('Factor type is incorrect.')
                
                # nomalize values for each posible state for message by subracting mean value of all state
                mean_value_for_all_state_of_message = np.mean(self.dic_factor_to_variable_message[(factor_idx, variable_idx)], axis=0)
                self.dic_factor_to_variable_message[(factor_idx, variable_idx)] -= mean_value_for_all_state_of_message

            # for all variable to factor messages
            for dic_key in self.dic_variable_to_factor_message.keys():

                # variable source, factor target
                variable_idx, factor_idx = dic_key

                self.dic_variable_to_factor_message[(variable_idx, factor_idx)] = np.zeros(shape=(self.num_state, 1))
                for neighbour_factor_i in self.dic_variable_neighbours[variable_idx]:
                    if neighbour_factor_i != factor_idx:
                        effect = 1
                        if self.factors[factor_idx]['type'] == 'pair':
                            effect = lambda_value
                        self.dic_variable_to_factor_message[(variable_idx, factor_idx)] += effect * self.dic_factor_to_variable_message[(neighbour_factor_i, variable_idx)]

                # nomalize values for each posible state for message by subracting mean value of all state
                mean_value_for_all_state_of_message = np.mean(self.dic_variable_to_factor_message[(variable_idx, factor_idx)], axis=0)
                self.dic_variable_to_factor_message[(variable_idx, factor_idx)] -= mean_value_for_all_state_of_message
        
        # calculate marginal for each variable
        marginals = np.zeros(shape=(self.num_state, self.num_variables))
        for [variable_i, neighbour_factors] in self.dic_variable_neighbours.items():
            
            for factor_idx in neighbour_factors:
                marginals[:, variable_i:(variable_i+1)] += self.dic_factor_to_variable_message[(factor_idx, variable_i)]

        # for each variable select state with highest value
        maximum_a_posteriori_estimation = np.argmax(marginals, axis=0)
        maximum_a_posteriori_estimation = np.reshape(maximum_a_posteriori_estimation, newshape=(self.height, self.width))

        return maximum_a_posteriori_estimation