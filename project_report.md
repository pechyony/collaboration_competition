[image1]: ./images/graph.jpg "convergence graph"  

# Project Report

In this project we controlled two agents to bounce a ball over net.

The agents were trained using multi-agent DDPG (MADDPG) algorithm. In the next two sections we describe this environment as well as the training process. 

## Tennis Environment

In [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment is provided by 
[Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 

## Training 2 Agents using Multi-Agent DDPG (MADDPG) Algorithm

We trained 2 agents using a [Multi-Agent Deterministic Policy Gradient (MADDPG) algorithm](https://arxiv.org/abs/1509.02971). MADDPG algorithm is an extension of single-agent [DDPG algorithm](https://arxiv.org/abs/1509.02971). In this section we describe various components of MADDPG and then we will present its complete pseudocode.

For each agent MADDPG algorithm maintains separate local and target actor functions a and a'. These functions are implemented as feedforward neural networks and have identical architecture. While actor networks of all agents have the same architecture, their weights are different across agents. The networks take as input agent's state vector S and output a vector of agent's continuous actions A. The dimensionality of A is the number of continuous actions performed by a single agent at every time point. We denote by A=a(S,w<sub>a</sub>) the vector of continuous actions at state S, computed by local actor neural network with weights w<sub>a</sub>. Similarly, A'=a'(S,w'<sub>a</sub>) is the vector of continuous actions at state S, computed by target actor neural network with weights w'<sub>a</sub>. We denote by a<sub>i</sub> and a'<sub>i</sub> local and target action functions of the i-th agent. We also denote by S<sup>i</sup> and A<sup>i</sup> state and action vectors of the i-th agent. Similarly, we denote by w<sub>a</sub><sup>i</sup> and w'<sub>a</sub><sup>i</sup> the weights of local and target actor networks of the i-th agent.

In addition to actor functions, for each agent MADDPG maintains separate local and target critic functions q and q'. Similarly to actor functions, critic functions are implemented as feedforward neural networks. Local and target critic networks have the same architecture, which is different from the architecture of actor networks. Similarly to actor networks, critic networks of all agents have the same architecture, but their weights are different across agents. The critic networks take as input a state vector S<sup>all</sup>=[S<sup>1</sup>,S<sup>2</sup>] and a vector of continuous actions A<sup>all</sup>=[A<sup>1</sup>,A<sup>2</sup>] of all agents. The output of critic network of i-th the agent is the value of actions of all agents A<sup>all</sup> for agent i, when they are taken at states S<sup>all</sup>. We denote by q(S,A,w<sub>c</sub>) the value of the vector of continuous actions A at state S, computed by local critic neural network with weights w<sub>c</sub>. Similarly, q'(S,A,w'<sub>c</sub>) is the value of continuous actions A at state S, computed by target critic neural network with weights w'<sub>c</sub>.

We denote by q<sub>i</sub> and q'<sub>i</sub> local and target critic functions of the i-th agent. Similarly, we denote by w<sub>c</sub><sup>i</sup> and w'<sub>c</sub><sup>i</sup> the weights of local and target critic networks of the i-th agent.
Let R<sup>i</sup> be the reward received by the i-th agent, S'<sup>i</sup> be the next state of the i-th agent and done<sup>i</sup> be an indicator if S'<sup>i</sup> is a final state.  

MADDPG uses a replay buffer that has a finite capacity and works in FIFO way. The replay buffer stores recent experiences of all agents and is a source of training data for actor and critic networks. A policy p(a(S,w<sub>a</sub>),&epsilon;,N) chooses a random action with probability &epsilon; and with probability 1-&epsilon; chooses an action a(S,w<sub>a</sub>) + N, where N is [Ornstein-Uhlenbeck noise process](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process). Each agent runs its own instance of the noise process.

We enhanced MADDPG algorithm with an exploration phase. In the first 1500 episodes the actions are chosen randomly, in the next 1500 episodes the random actions and actions from actor networks are chosen with equal probability 0.5. Only after 3000 episodes actor networks are responsible for generating all actions. Notice that when taking random actions the agents still learn from the resulting experiences. This extensive exploration step was necessary to solve Tennis environment using MADDPG.

Our implementation of MADDPG algorithm is summarized below:

Initialize replay buffer with capacity BUFFER_SIZE  
For agent i=1,2, initialize local actor function a<sub>i</sub> with weights w<sub>a</sub>  
For agent i=1,2, initialize target actor function a'<sub>i</sub> with weights w'<sub>a</sub> = w<sub>a</sub>  
Initialize local critic function q with weights w<sub>c</sub>  
Initialize target critic function q' with weights w'<sub>c</sub> = w<sub>c</sub>  
Set &epsilon;=1   
Initialize number of episodes by 0  

While environment is not solved:   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Increase the number of episode by 1    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Get from the environment initial state S<sup>i</sup>, i=1,...,2 of every agent      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While all of done<sup>i</sup>, i=1,2, are false:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For agent i=1,2:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
If number of episode < PURE_EXPLORATION   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choose a random action A<sup>i</sup>    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else if number of episode < MIXED_EXPLORATION     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   choose random action A<sup>i</sup> with probability 0.5, choose action A<sup>i</sup> using policy p(a<sub>i</sub>(S<sup>i</sup>,w<sub>a</sub><sup>i</sup>),&epsilon;,N)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choose action A<sup>i</sup> using policy p(a<sub>i</sub>(S<sup>i</sup>,w<sub>a</sub><sup>i</sup>),&epsilon;,N)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Take action A<sup>i</sup>, observe reward R<sup>i</sup>, next state S'<sup>i</sup> and the indicator done<sup>i</sup>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;S<sup>i</sup> = S'<sup>i</sup>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store experience tuple (S<sup>1</sup>,A<sup>1</sup>,R<sup>1</sup>,S'<sup>1</sup>,done<sup>1</sup>,S<sup>2</sup>,A<sup>2</sup>,R<sup>2</sup>,S'<sup>2</sup>,done<sup>2</sup>) in the replay buffer    
    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a batch of K experiences (S<sub>j</sub><sup>1</sup>,A<sub>j</sub><sup>1</sup>,R<sub>j</sub><sup>1</sup>,S'<sub>j</sub><sup>1</sup>,done<sub>j</sub><sup>1</sup>,S<sub>j</sub><sup>2</sup>,A<sub>j</sub><sup>2</sup>,R<sub>j</sub><sup>2</sup>,S'<sub>j</sub><sup>2</sup>,done<sub>j</sub><sup>2</sup>), j=1 ... K, from the replay buffer    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For i = 1,2:   
    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Get next-state actions from actor target network   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A'<sub>j</sub><sup>1</sup>=a'(S'<sub>j</sub><sup>1</sup>,w'<sub>a</sub><sup>1</sup>), j=1,...,K     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A'<sub>j</sub><sup>2</sup>=a'(S'<sub>j</sub><sup>2</sup>,w'<sub>a</sub><sup>2</sup>), j=1,...,K

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Get next-state action values from critic target network  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q'<sub>j</sub><sup>i</sup>=q'<sup>i</sup>([S'<sub>j</sub><sup>1</sup>,S'<sub>j</sub><sup>2</sup>], [A'<sub>j</sub><sup>1</sup>,A'<sub>j</sub><sup>2</sup>], w'<sub>c</sub><sup>i</sup>), j=1,...,K   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
// Compute state-action value targets for current states   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q<sub>j</sub><sup>i</sup>=r<sub>j</sub><sup>i</sup>+&gamma;&middot;Q'<sub>j</sub><sup>i</sup>&middot;(1-done<sub>j</sub><sup>i</sup>), j=1,...,K   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
// Update local critic network   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &Delta;w<sub>c</sub><sup>i</sup>=&sum;<sub>j=1 ... K</sub> (Q<sub>j</sub><sup>i</sup>-q<sub>i</sub>([S<sub>j</sub><sup>1</sup>,S<sub>j</sub><sup>2</sup>], [A<sub>j</sub><sup>1</sup>,A<sub>j</sub><sup>2</sup>], w<sub>c</sub><sup>i</sup>)) &middot; &nabla;q([S<sub>j</sub><sup>1</sup>,S<sub>j</sub><sup>2</sup>], [A<sub>j</sub><sup>1</sup>,A<sub>j</sub><sup>2</sup>], w<sub>c</sub><sup>i</sup>)    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w<sub>c</sub><sup>i</sup> = w<sub>c</sub><sup>i</sup> - &alpha;<sub>c</sub>&middot;&Delta;w<sub>c</sub><sup>i</sup>   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Update local actor network   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let dq<sub>i</sub>([S<sub>j</sub><sup>1</sup>,S<sub>j</sub><sup>2</sup>], [a<sub>1</sub>(S<sub>j</sub><sup>1</sup>,w<sub>a</sub><sup>1</sup>), a<sub>2</sub>(S<sub>j</sub><sup>2</sup>,w<sub>a</sub><sup>2</sup>)], w<sub>c</sub><sup>i</sup>) be a derivative of q<sub>i</sub>([S<sub>j</sub><sup>1</sup>,S<sub>j</sub><sup>2</sup>], [a<sub>1</sub>(S<sub>j</sub><sup>1</sup>,w<sub>a</sub><sup>1</sup>), a<sub>2</sub>(S<sub>j</sub><sup>2</sup>,w<sub>a</sub><sup>2</sup>)], w<sub>c</sub><sup>i</sup>) with respect to a<sub>i</sub>(S<sub>j</sub><sup>i</sup>,w<sub>a</sub><sup>i</sup>)        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &Delta;w<sub>a</sub><sup>i</sup>=&sum;<sub>i=1 ... K</sub> -q<sub>i</sub>([S<sub>j</sub><sup>1</sup>,S<sub>j</sub><sup>2</sup>], [a<sub>1</sub>(S<sub>j</sub><sup>1</sup>,w<sub>a</sub><sup>1</sup>), a<sub>2</sub>(S<sub>j</sub><sup>2</sup>,w<sub>a</sub><sup>2</sup>)], w<sub>c</sub><sup>i</sup>)&middot;&nabla;a(S<sub>i</sub>,w<sub>a</sub>)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w<sub>a</sub><sup>i</sup> = w<sub>a</sub><sup>i</sup> - &alpha;<sub>a</sub>&middot;&Delta;w<sub>a</sub><sup>i</sup>   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Update target networks  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w'<sub>a</sub>=&tau;&middot;w<sub>a</sub><sup>i</sup>+(1-&tau;)&middot;w'<sub>a</sub><sup>i</sup>   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w'<sub>c</sub>=&tau;&middot;w<sub>c</sub><sup>i</sup>+(1-&tau;)&middot;w'<sub>c</sub><sup>i</sup>  

The following table lists the architecture of local and target actor networks.

| Name | Type | Input | Output | Activation function | 
|:-:|:-----:|:-----:|:------:|:-------------------:|  
|Input Layer| Fully Connected  | 24    | 256    | ReLU                |
|Hidden Layer| Fully Connected   | 256    | 128     | ReLU                |
|Output Layer| Fully Connected   | 128    | 2      |  tanh             |

The following table lists the architecture of local and target critic networks.

| Name | Type | Input | Output | Activation function | 
|:-:|:-----:|:-----:|:------:|:-------------------:|  
|Input Layer| Fully Connected  | 52    | 256    |  Leaky ReLU                |
|Hidden Layer| Fully Connected   | 256    | 128     | Leaky ReLU                |
|Output Layer| Fully Connected   | 128    | 1      | Leaky ReLU        |

The weights in the first two layers were initialized using Glorot initialization. The weights from the last layer were initialized from a uniform distribution [-1e-3,1e-3]. We tuned the values of hyperparameters to solve the environment. The next table summarizes the our final values of hyperparameters:

| Hyperparameter | Description | Value |
|:--------------:|:-----------:|:-----:|
| BUFFER_SIZE | Size of replay buffer | 1000000 |
| K | Batch size | 256 |
| &gamma; | Discount factor | 0.99 |
| &alpha;<sub>a</sub> | Learning rate of local actor network | 0.0001 |
| &alpha;<sub>c</sub> | Learning rate of local critic network | 0.001 | 
| &tau; | Weight of local network when updating target network| 0.001 |
| PURE_EXPLORATION | Number of episodes when only random actions are executed | 1500 |  
| MIXED_EXPLORATION | Number of episodes when random actions are executed with probability 0.5 | 1500 |
| &mu; | mean value of Ornstein-Unlenbeck process | 0 |
| &theta; | growth rate of Ornstein-Unlenbeck process | 0.15 |
| &sigma; | standard deviation of Ornstein-Unlenbeck process | 0.2 |

## Software packages

We used PyTorch to train neural network and  the API of Unity Machine Learning Agents Toolkit to interact with Reacher environment. 

## Results

MADDPG algorithm solves environment in 10504 episodes. The following graph shows the scores of individual episodes as a function of the number of episode.  

![alt text][image1]

## Ideas for Future Work

While we were able to solve the environment, the convergence is slow. We tuned several hyperparameters (e.g. learning rate of actor and critic networks, type of activation units and number of activation units in each layer). However additional tuning of hyperparameter is needed to speed up the convergence. We plan to try the following ideas:
* use batch normalization
* use prioritized replay
* use weight decays in actor and critic networks 

After improving convergence rate in Tennis environment, we plan to try more challenging [Soccer environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). 

