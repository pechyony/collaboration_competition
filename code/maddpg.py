# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor, transpose_list
from multiagent_algorithm import MultiAgentAlgorithm
from buffer import ReplayBuffer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters of MADDPG algorithm
BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 256         # minibatch size 256
GAMMA = 0.99             # discount factor w
TAU = 0.001              # for soft update of target parameters
ACTOR_FC1_UNITS = 256    # number of neurons in the first hidden layer of actor network 
ACTOR_FC2_UNITS = 128    # number of neurons in the second hidden layer of actor network
CRITIC_FC1_UNITS = 256   # number of neurons in the first hidden layer of critic network
CRITIC_FC2_UNITS = 128   # number of neurons in the second hidden layer of critic network
LR_ACTOR = 1e-4          # learning rate of actor
LR_CRITIC = 1e-3         # learning rate of critic
WEIGHT_DECAY_ACTOR = 0   # weight decay of actor
WEIGHT_DECAY_CRITIC = 0  # weight decay of critic
UPDATE_EVERY = 1         # number of steps between every round of updates
N_UPDATES = 1            # number of batches in a single round of updates


class MADDPG(MultiAgentAlgorithm):
    def __init__(self, action_size, n_agents, seed, state_size):
        super().__init__(action_size, n_agents, seed)

        # critic input = obs_full + actions = 14+2+2+2=20
        self.agents = [DDPGAgent(state_size, ACTOR_FC1_UNITS, ACTOR_FC2_UNITS, action_size, 
                                 (state_size+action_size)*n_agents, CRITIC_FC1_UNITS, CRITIC_FC2_UNITS, 
                                 LR_ACTOR, LR_CRITIC, WEIGHT_DECAY_ACTOR, WEIGHT_DECAY_CRITIC) for i in range(n_agents)]
        self.n_agents = n_agents
        self.epsilon = 0
        self.iter = 0
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
    def save_model(self, model_file):
        """Save networks and all other model parameters
        
        Params
        ======
            model_file (string): name of the file that will store the model
        """
        checkpoint = {'actor_local1': self.agents[0].actor.state_dict(),
                      'critic_local1': self.agents[0].critic.state_dict(),
                      'actor_target1': self.agents[0].target_actor.state_dict(),
                      'critic_target1': self.agents[0].target_critic.state_dict(),
                      'actor_local2': self.agents[1].actor.state_dict(),
                      'critic_local2': self.agents[1].critic.state_dict(),
                      'actor_target2': self.agents[1].target_actor.state_dict(),
                      'critic_target2': self.agents[1].target_critic.state_dict()}
        
        torch.save(checkpoint, model_file)
    
    def load_model(self, model_file):
        """Load networks and all other model parameters
        
        Params
        ======
            model_file (string): name of the file that stores the model
        """
        checkpoint = torch.load(model_file)
        self.agents[0].actor.load_state_dict(checkpoint['actor_local1'])
        self.agents[0].critic.load_state_dict(checkpoint['critic_local1'])
        self.agents[0].target_actor.load_state_dict(checkpoint['actor_target1'])
        self.agents[0].target_critic.load_state_dict(checkpoint['critic_target1'])
        self.agents[1].actor.load_state_dict(checkpoint['actor_local2'])
        self.agents[1].critic.load_state_dict(checkpoint['critic_local2'])
        self.agents[1].target_actor.load_state_dict(checkpoint['actor_target2'])
        self.agents[1].target_critic.load_state_dict(checkpoint['critic_target2'])

    def act(self, states):
        """get actions from all agents in the MADDPG object"""
        
        actions = []
        for agent, state in zip(self.agents, states):
            if np.random.rand() < self.epsilon:
                actions_agent = np.random.randn(2) 
                actions_agent = np.clip(actions_agent, -1, 1)
                actions.append(actions_agent)
            else:
                actions.append(agent.act(state))
        return actions

    def target_act(self, states):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs) for agent, obs in zip(self.agents, states)]
        return target_actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.

        Params
        ======
            states (array_like): current state (for each agent)
            actions (array_like): action taken at the current state (for each agent) 
            rewards (array_like): reward from an action (for each agent)
            next_states (array_like): next state of environment (for each agent)
            dones (array_like): true if the next state is the final one, false otherwise (for each agent)
        """

        # Save experience / reward
        self.buffer.add(states, actions, rewards, next_states, dones)

        self.iter = (self.iter + 1) % UPDATE_EVERY
        if self.iter == 0:    

            # Learn, if enough samples are available in buffer
            if len(self.buffer) > BATCH_SIZE:
                for i in range(N_UPDATES):
                    experiences = self.buffer.sample()
                    for agent in range(self.n_agents):
                        self.learn(experiences, agent)
                        self.update_targets(agent)

    def learn(self, experiences, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        
        states, actions, rewards, next_states, dones = experiences

        agent = self.agents[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network

        target_actions = self.target_act(next_states)
        target_actions = torch.cat(target_actions, dim=1)
        t = torch.tensor(transpose_list(next_states.cpu().data.numpy()))
        next_states_all = t.view(t.shape[0],-1).to('cpu')
        target_critic_input = torch.cat((next_states_all,target_actions.to('cpu')), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = rewards[agent_number].view(-1, 1) + GAMMA * q_next * (1 - dones[agent_number].view(-1, 1))
        actions_all = torch.cat(torch.unbind(actions), dim=1)
        t = torch.tensor(transpose_list(states.cpu().data.numpy()))
        states_all = t.view(t.shape[0],-1).to('cpu')
        critic_input = torch.cat((states_all, actions_all.to('cpu')), dim=1).to(device)
        q = agent.critic(critic_input)

        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward(retain_graph=True)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.agents[i].actor(state) if i == agent_number \
                   else self.agents[i].actor(state).detach()
                   for i, state in enumerate(states)]
        q_input = torch.cat(q_input, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((states_all.to('cpu'), q_input.to('cpu')), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward(retain_graph=True)
        agent.actor_optimizer.step()

    def update_targets(self, i):
        """soft update targets"""
        soft_update(self.agents[i].target_actor, self.agents[i].actor, TAU)
        soft_update(self.agents[i].target_critic, self.agents[i].critic, TAU)
            
            
            




