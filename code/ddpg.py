# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, in_actor, actor_fc1_units, actor_fc2_units, out_actor, 
                 in_critic, critic_fc1_units, critic_fc2_units, lr_actor, lr_critic, 
                 weight_decay_actor, weight_decay_critic):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(in_actor, actor_fc1_units, actor_fc2_units, out_actor).to(device)
        self.critic = Critic(in_critic, critic_fc1_units, critic_fc2_units, 1).to(device)
        self.target_actor = Actor(in_actor, actor_fc1_units, actor_fc2_units, out_actor).to(device)
        self.target_critic = Critic(in_critic, critic_fc1_units, critic_fc2_units, 1).to(device)

        self.target_actor.eval()
        self.target_critic.eval()

        self.noise = OUNoise(out_actor)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)


    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state) 
        self.actor.train()

        # add exploration noise to the action
        action += self.noise.noise()
        
        return np.clip(action.cpu().data.numpy(), -1, 1)

    def target_act(self, state):
        with torch.no_grad():
            action = self.target_actor(state) 
    
        return action
