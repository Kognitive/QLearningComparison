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
        return self.choose_action(q, config)
