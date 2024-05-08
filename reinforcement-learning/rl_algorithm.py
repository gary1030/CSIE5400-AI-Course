import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined
        # define the network

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        # forward pass
        
        return x

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        epsilon=0.9,
        epsilon_min=0.05,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        """
        DQN agent has four methods.

        - __init__() is the standard initializer.
        - act(), which receives a state as an np.ndarray and outputs actions following the epsilon-greedy policy.
        - process(), which processes a single transition and defines the agent's actions at each step.
        - learn(), which samples a mini-batch from the replay buffer to train the Q-network.
        """
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        "*** YOUR CODE HERE ***"
        # define the network, target network, optimizer, buffer, total_steps, epsilon_decay
        utils.raiseNotDefined()
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            "*** YOUR CODE HERE ***"
            utils.raiseNotDefined()
            # Random action
            
        else:
            "*** YOUR CODE HERE ***"
            utils.raiseNotDefined()
            # output actions by following epsilon-greedy policy
        
        return 0
    
    def learn(self):
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        # samples a mini-batch from replay buffer and train q-network
        
        return {} # return the information you need for logging
    
    def process(self, transition):
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        # takes one transition as input and define what the agent do for each step.
        
        
        return {} # return the information you need for logging