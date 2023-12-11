
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from framework.BaseAgent import BaseAgent

class PolicyNetwork(nn.Module):
    """
    A Simple Policy Network to estimate the policy of the agent given a state.
    """

    def __init__(self, actionSpace):
        super().__init__()

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, actionSpace)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(hid)
        return F.softmax(self.fc3(hid), dim=-1)
    

class PolicyGradientAgentMC(BaseAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionSpace, gamma=0.99, lr=0.001):
        super().__init__(actionSpace)
        
        self.gamma = gamma

        self.policy = PolicyNetwork(actionSpace)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)

        self.logProbs = []
        self.reward = []
        self.finalReward = []

    def act(self, state: torch.Tensor, stage: dict = None) -> torch.Tensor:
        actionProb = self.policy(state)
        actionDist = Categorical(actionProb)
        action = actionDist.sample()
        logProb = actionDist.log_prob(action)

        self.lastAction = {"action": action, "logProb": logProb, "actionProb": actionProb}

        return self.lastAction
    
    def learn(self, 
              state: torch.Tensor, 
              action: torch.Tensor, 
              reward: torch.Tensor, 
              nextState: torch.Tensor, 
              done: int, 
              stage: dict)->dict:
        """ 
        Collect the reward and log_prob of the action.
        """

        self.logProbs.append(self.lastAction['logProb'])
        self.reward.append(reward)
        if done or stage['step'] == stage['maxStep'] - 1:
            self.finalReward.append(reward)
        
        return {}

    def onTrainEpisodeDone(self, stage: dict):
        """ 
        Learn from the collected experience.
        This function will only run once every episode. 
        
        return: a dictionary of meta data {'key': value} to be recorded.
        """
        
        return {}
    
    def onTrainBatchDone(self, stage: dict):

        reward = torch.cat(self.reward, dim=0)

        avgReward = reward.sum().item()/len(self.finalReward)
        finalReward = torch.cat(self.finalReward, dim=0).mean().item()

        # monte carlo method
        for i in np.arange(reward.size(0)-2, -1, -1):
            reward[i] += self.gamma * reward[i+1]

        # normalize reward
        reward = (reward - torch.mean(reward)) / (torch.std(reward) + 1e-9)

        logProbs = torch.stack(self.logProbs)

        # train policy network
        loss = (-logProbs * reward).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logProbs = []
        self.reward = []
        self.finalReward = []

        return {'loss': loss.item(), 'avgReward': avgReward, 'finalReward': finalReward}