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
        self.hidden_size = self.config['num_hidden']
        self.model_range = tf.range(0, self.config['num_models'], 1)

        # obtain sizes of model
        self.lss = config['state_space'].get_log2_size()
        self.las = config['action_space'].get_log2_size()
        self.D = self.lss + self.las

        # first of all initialize the weights
        self.weights = self.init_weights(self.num_models, self.D, self.hidden_size)

    def get_graph(self, states: tf.Tensor, actions: tf.Tensor):
        """This method delivers the graph to the outside.

        Args:
            states: The states to use as an input
            actions: The corresponding actions to use.
        """

        # convert actions and states
        binary_states = self.int_to_binary(self.lss, states)
        binary_actions = self.int_to_binary(self.las, actions)
        conc = tf.cast(tf.concat([binary_states, binary_actions], axis=0), tf.float32)

        # access the weights one by one
        [W, V, b, c] = self.weights

        # first of all create a diagonal matrix mask for each column vector of self.v
        weight_masked_matrix = tf.multiply(W, tf.expand_dims(tf.transpose(conc), 1))
        hidden_layer = tf.sigmoid(tf.cumsum(weight_masked_matrix, axis=2, exclusive=True) + c)

        # calc the distribution
        pre_p_dist = tf.einsum('mdh,mhd->dm', V, hidden_layer) + tf.transpose(b)
        p_dist = tf.sigmoid(pre_p_dist)

        # one computational graph improvement
        inv_v = tf.constant(1.0) - conc
        inv_p_dist = tf.constant(1.0) - p_dist

        # get log values
        log_iv_p_dist = tf.nn.softplus(pre_p_dist)
        log_p_dist = tf.nn.softplus(-pre_p_dist)

        # this gets the evaluation graph, if only one sample is supplied
        # self.evaluation_model = tf.reduce_prod(tf.pow(p_dist, self.v) + tf.pow(inv_p_dist, inv_v), axis=0)
        evaluation_model = tf.reduce_prod(tf.multiply(p_dist, conc) + tf.multiply(inv_p_dist, inv_v), axis=0)
        nll = -tf.reduce_sum(-conc * log_p_dist - inv_v * log_iv_p_dist)

        # Place for the saved evaluation
        saved_evaluation = tf.get_variable("saved_evaluation", [self.num_models], trainable=False)
        save_evaluation = tf.assign(saved_evaluation, evaluation_model)

        # retrieve the minimizer
        with tf.control_dependencies([save_evaluation]):
            minimizer = tf.train.AdamOptimizer(0.0001).minimize(nll, var_list=[W, V, b, c])

        # execute the minimizer in prior
        with tf.control_dependencies([minimizer]):
            pseudo_counts = saved_evaluation * (1 - evaluation_model) / (evaluation_model - saved_evaluation)

        # create normal count based update
        cb_step_value = tf.get_variable("cb_step", dtype=tf.int64, shape=[], initializer=tf.zeros_initializer)
        cb_step_update = tf.count_up_to(cb_step_value, 2 ** 63 - 1)

        # group both actions
        cb_update = tf.group(cb_step_update)

        # return the model
        return pseudo_counts, cb_step_value, cb_update

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

    def init_weights(self, num_models: int, input_size: int, hidden_size: int):
        """This method initializes the weights for all density models.

        Args:
            num_models: The number of different models
            input_size: The input dimension to this problem
            hidden_size: The number of neurons used inside of the hidden layer.
        """

        # create the vectors for V and W
        W = self.init_single_weight([num_models, hidden_size, input_size], "W")
        V = self.init_single_weight([num_models, input_size, hidden_size], "V")
        b = self.init_single_weight([num_models, input_size], "b")
        c = self.init_single_weight([num_models, hidden_size, 1], "c")

        # return all weights
        return [W, V, b, c]

    def init_single_weight(self, size, name):
        """This method initializes a single weight. The only thing you have
        to specify is the size of the tensor as well as the name.

        Args:
            size: The shape of the weight.
            name: The name of this tensor
        """

        return tf.Variable(tf.random_normal(size, mean=0.0, stddev=0.01), name=name)