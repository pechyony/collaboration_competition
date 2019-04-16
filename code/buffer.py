import numpy as np
from collections import namedtuple, deque
import random
import torch
from utilities import transpose_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        Params
        ======
            state (array_like): current of an agent
            action (float): action taken at the current state
            reward (float): reward from an action
            next_state (array_like): next state of an agent
            dones (boolean): true if the next state is the final one, false otherwise (for each agent)

        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array(transpose_list([e.state for e in experiences if e is not None]))).float().to(device)
        actions = torch.from_numpy(np.array(transpose_list([e.action for e in experiences if e is not None]))).float().to(device)
        rewards = torch.from_numpy(np.array(transpose_list([e.reward for e in experiences if e is not None]))).float().to(device)
        next_states = torch.from_numpy(np.array(transpose_list([e.next_state for e in experiences if e is not None]))).float().to(device)
        dones = torch.from_numpy(np.array(transpose_list([e.done for e in experiences if e is not None])).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


