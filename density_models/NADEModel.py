import tensorflow as tf


class NADEModel:
    """Constructs a NADE Density Model."""

    def __init__(self, config: dict):
        """Creates a new NADEModel using the supplied configuration

        Args:
            config:
                num_models: The number of independent density models to create.
                action_space: The action space to use for the model.
                state_space: The state space to use for the model.
            """

        # save sizes of model
        self.config = config
        self.num_models = self.config['num_models']
        self.num_heads = self.config['num_heads']
        self.heads_per_sample = self.config['heads_per_sample']
        self.hidden_size = self.config['num_hidden']
        self.model_range = tf.range(0, self.config['num_models'], 1)

        # obtain sizes of model
        self.ss = config['state_space'].get_size()
        self.acts = config['action_space'].get_size()
        self.lss = config['state_space'].get_log2_size()
        self.las = config['action_space'].get_log2_size()
        self.D = self.lss + self.las

        # first of all initialize the weights
        self.weights = self.init_weights(self.num_models, self.num_heads, self.D, self.hidden_size)

    def get_graph(self, states: tf.Tensor, actions: tf.Tensor, head_mask_tensor: tf.Tensor, minimizer: bool):
        """This method delivers the graph to the outside.

        Args:
            states: The states to use as an input
            actions: The corresponding actions to use.
        """

        # convert actions and states
        binary_states = self.int_to_binary(self.lss, states)
        binary_actions = self.int_to_binary(self.las, actions)
        conc = tf.cast(tf.concat([binary_states, binary_actions], axis=0), tf.float64)

        # access the weights one by one
        [W, V, b, c] = self.weights

        # first of all create a diagonal matrix mask for each column vector of self.v
        expanded_mult_conc = tf.expand_dims(tf.expand_dims(tf.transpose(conc), 1), 1)
        weight_masked_matrix = tf.multiply(W, expanded_mult_conc)
        hidden_layer = tf.sigmoid(tf.cumsum(weight_masked_matrix, axis=3, exclusive=True) + c)

        # calc the distribution
        calc_tensor = tf.einsum('mkdh,mkhd->dmk', V, hidden_layer)
        pre_p_dist = calc_tensor + tf.transpose(b, [2, 0, 1])
        p_dist = tf.sigmoid(pre_p_dist)

        # one computational graph improvement
        exp_conc = tf.expand_dims(conc, 2)
        inv_v = tf.constant(1.0, dtype=tf.float64) - exp_conc
        inv_p_dist = tf.constant(1.0, dtype=tf.float64) - p_dist

        # this gets the evaluation graph, if only one sample is supplied
        # self.evaluation_model = tf.reduce_prod(tf.pow(p_dist, self.v) + tf.pow(inv_p_dist, inv_v), axis=0)
        density_value = tf.reduce_prod(tf.multiply(p_dist, exp_conc) + tf.multiply(inv_p_dist, inv_v), axis=0)

        # when a minimizer is needed as well
        if minimizer:

            # get log values
            log_iv_p_dist = tf.nn.softplus(pre_p_dist)
            log_p_dist = tf.nn.softplus(-pre_p_dist)

            nll = -tf.reduce_sum(tf.expand_dims(tf.cast(head_mask_tensor, tf.float64), 0) * (-exp_conc * log_p_dist - inv_v * log_iv_p_dist))
            minimizer = tf.train.AdamOptimizer(self.config['learning_rate']).minimize(nll, var_list=[W, V, b, c])

            # return the model
            return density_value, minimizer

        else:
            return density_value

    def get_all_densities(self):

        state_vector = self.int_to_binary(self.lss, tf.range(0, self.ss, 1, dtype=tf.int64))
        action_vector = self.int_to_binary(self.las, tf.range(0, self.acts, 1, dtype=tf.int64))

        # create the matrix containing all
        single_action_list = list()
        for ae in range(self.acts):
            current_action_vector = tf.expand_dims(action_vector[:, ae], 1)
            current_tiled_action_matrix = tf.tile(current_action_vector, [1, self.ss])
            single_action_list.append(tf.cast(tf.concat([state_vector, current_tiled_action_matrix], axis=0), tf.float64))

        # concatenate entlong the sample dimension
        conc = tf.concat(single_action_list, axis=1)

        # access the weights one by one
        [W, V, b, c] = self.weights

        # first of all create a diagonal matrix mask for each column vector of self.v
        weight_masked_matrix = tf.einsum('mkhd,dn->mkhdn', W, conc)
        hidden_layer = tf.sigmoid(tf.cumsum(weight_masked_matrix, axis=3, exclusive=True) + tf.expand_dims(c, 4))

        # calc the distribution
        pre_p_dist = tf.einsum('mkdh,mkhdn->mkdn', V, hidden_layer) + tf.expand_dims(b, 3)
        p_dist = tf.sigmoid(pre_p_dist)

        # one computational graph improvement
        exp_conc = tf.expand_dims(tf.expand_dims(conc, 0), 0)
        inv_v = tf.constant(1.0, dtype=tf.float64) - exp_conc
        inv_p_dist = tf.constant(1.0, dtype=tf.float64) - p_dist

        # this gets the evaluation graph, if only one sample is supplied
        pre_density_value = tf.reduce_prod(tf.multiply(p_dist, exp_conc) + tf.multiply(inv_p_dist, inv_v), axis=2)
        all_density_values = tf.transpose(tf.reshape(pre_density_value, [self.num_models, self.num_heads, self.acts, self.ss]), [0, 1, 3, 2])

        return all_density_values

    def int_to_binary(self, num_bits: int, input: tf.Tensor) -> tf.Tensor:
        """This method converts an input number to a tensor
        with the corresponding binary values.

        Args:
            num_bits: The number of bits the number should be represented.
            input: The input itself, should be a tensor of rank 1 where the size is equal to num_models"""

        # create the conversion graph
        vector = input
        binary_vec_lst = list()
        for i in range(num_bits):
            binary_vec_lst.append(tf.floormod(vector, 2))
            vector = tf.floordiv(vector, 2)

        binary_values = tf.stack(binary_vec_lst, axis=0)

        # pass back the result
        return binary_values

    def init_weights(self, num_models: int, num_heads: int, input_size: int, hidden_size: int):
        """This method initializes the weights for all density models.

        Args:
            num_models: The number of different models
            num_heads: The number of different heads
            input_size: The input dimension to this problem
            hidden_size: The number of neurons used inside of the hidden layer.
        """

        # create the vectors for V and W
        W = self.init_single_weight([num_models, num_heads, hidden_size, input_size], "W")
        V = self.init_single_weight([num_models, num_heads, input_size, hidden_size], "V")
        b = self.init_single_weight([num_models, num_heads, input_size], "b")
        c = self.init_single_weight([num_models, num_heads, hidden_size, 1], "c")

        # return all weights
        return [W, V, b, c]

    def init_single_weight(self, size, name):
        """This method initializes a single weight. The only thing you have
        to specify is the size of the tensor as well as the name.

        Args:
            size: The shape of the weight.
            name: The name of this tensor
        """

        return tf.Variable(tf.random_normal(size, mean=0.0, stddev=0.01, dtype=tf.float64), name=name, dtype=tf.float64)