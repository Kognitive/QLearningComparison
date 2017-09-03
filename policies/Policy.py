class Policy:
    """Represents a basic policies_nn. It therefore needs a function, which
    evaluates successfully the q function"""

    def choose_action(self, q, config):
        """This method of a policies_nn basically gets a Q function
        and has to return the action to take now. Here you can
        specify behaviour like taking the best action, or sometimes
        different actions to them.

        Args:
            q: The q function to use for evaluating.

        Returns: The index of the action that should be taken
        """
        raise NotImplementedError("Please implement select_action method")

    def select_action(self, q, config):
        """This method of a policies_nn basically gets a Q function
        and has to return the action to take now. Here you can
        specify behaviour like taking the best action, or sometimes
        different actions to them.

        Args:
            q: The q function to use for evaluating.

        Returns: The index of the action that should be taken
        """

        # modulate q
        mod_q = q + (config['ucb_term'] if config is not None and 'ucb' in config and config['ucb'] in config else 0)
        mod_q += (config['pseudo-count'] if config is not None and 'pseudo-count' in config and config['pseudo-count'] in config else 0)
        return self.choose_action(q, config)
