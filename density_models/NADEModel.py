import tensorflow as tf


class NADEModel:
    """Constructs a NADE Density Model."""

    def __init__(self, config):
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

        # define the variables and placeholders
        self.v = tf.placeholder(tf.float32, [self.D, self.num_models], name="v")

        # first of all initialize the weights
        [self.W, self.V, self.b, self.c] = self.init_weights(self.D, self.hidden_size)

        # --------------- GRAPH ----------------------

        # first of all create a diagonal matrix mask for each column vector of self.v
        weight_masked_matrix = tf.einsum('ij,kjl->kil', self.W, tf.matrix_diag(tf.transpose(self.v)))
        hidden_layer = tf.sigmoid(tf.cumsum(weight_masked_matrix, axis=2, exclusive=True) + self.c)

        # calc the distribution
        pre_p_dist = tf.einsum('ij,kji->ik', self.V, hidden_layer) + self.b
        p_dist = tf.sigmoid(pre_p_dist)

        # one computational graph improvement
        inv_v = tf.constant(1.0) - self.v
        inv_p_dist = tf.constant(1.0) - p_dist

        # get log values
        log_iv_p_dist = tf.nn.softplus(pre_p_dist)
        log_p_dist = tf.nn.softplus(-pre_p_dist)

        # this gets the evaluation graph, if only one sample is supplied
        #self.evaluation_model = tf.reduce_prod(tf.pow(p_dist, self.v) + tf.pow(inv_p_dist, inv_v), axis=0)
        self.evaluation_model = tf.reduce_prod(tf.multiply(p_dist, self.v) + tf.multiply(inv_p_dist, inv_v), axis=0)
        self.nll = -tf.reduce_mean(tf.reduce_sum(-self.v * log_p_dist - inv_v * log_iv_p_dist, axis=0))

        # retrieve the minimizer
        self.minimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.nll, var_list=[self.W, self.V, self.b, self.c])

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_graph(self, states, actions):

    # inits the weights
    def init_weights(self, input_size, hidden_size):

        # create the vectors for V and W
        W = self.init_single_weight([hidden_size, input_size], "W")
        V = self.init_single_weight([input_size, hidden_size], "V")
        b = self.init_single_weight([input_size, 1], "b")
        c = self.init_single_weight([hidden_size, 1], "c")

        # return all weights
        return [W, V, b, c]

    # inits the weights
    def init_single_weight(self, size, name):
        return tf.Variable(tf.random_normal(size, mean=0.0, stddev=0.01), name=name)

    # this method basically walks one step in the direction
    # for the
    def step(self, samples, num_steps):

        # when the size of the memory is zero use the original samples
        if (self.MS == 0):

            # simply use the passed samples
            rand_samples = samples

        else:

            # length
            N = np.size(samples, 1)

            # add them to the memory
            for i in range(N):
                self.M.insert(samples[:, i])

            # sample a minibatch of same length than samples
            rand_samples = self.M.sample(10)


        for k in range(num_steps):

            # print("Training batch is: ", np.transpose(samples))
            self.sess.run(self.minimizer, feed_dict={ self.v: rand_samples })

    # this method actually calculates the log likelihood
    def get_log_likelihood(self, samples):
        return self.sess.run(self.nll, feed_dict={ self.v: samples })

    # evaluates the model at sample
    def evaluate(self, sample):
        val = self.sess.run(self.evaluation_model, feed_dict={ self.v: sample })
        return val