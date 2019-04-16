from abc import ABCMeta, abstractmethod
import numpy as np

class MultiAgentAlgorithm(metaclass=ABCMeta):
    """Interacts with and learns from environment by running multiple agents"""

    def __init__(self, action_size, n_agents, seed):
        """Initializes a MultiAgentAlgorithm object.

        Params
        ======
            action_size (int): number of possible actions
            n_agents (int): number of agents
            seed (int): random seed
        """
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = np.random.seed(seed)

    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        """Learns from interaction with environment.

        Params
        ======
            states (array_like): current state of each agent
            actions (array_like): action taken at the current state of each agent 
            rewards (array_like): reward from an action taken by each agent
            next_states (array_like): next state of environment of each agent
            dones (array_like): for each agent, true if the next state is the final one, false otherwise
        """
        pass


    @abstractmethod
    def act(self, state):
        """Finds the best action for a given observation of each agent.

        Params
        ======
            state (array_like): current state of each agent

        Returns
        =======
            actions (array_like): action that should be taken at the current state of each agent
        """
        pass

