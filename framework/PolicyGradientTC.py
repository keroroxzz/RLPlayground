
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

        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, actionSpace)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(hid)
        return F.softmax(self.fc3(hid), dim=-1)
    

class PolicyGradientAgentTC(BaseAgent):
    """
    A Policy Gradient Agent that uses Temporal method to estimate the policy gradient.
    """

    def __init__(self, actionSpace, gamma=0.99, lr=0.001):
        super().__init__(actionSpace)
        
        self.gamma = gamma

        self.policy = PolicyNetwork(actionSpace)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)

        self.losses = []
        self.reward = []
        self.finalReward = []

    def act(self, state: torch.Tensor, stage: dict = None) -> torch.Tensor:
        actionReward = self.policy(state)

        actionDist = Categorical(torch.softmax(actionReward, dim=-1))
        action = actionDist.sample()
        logProb = actionDist.log_prob(action)

        self.lastAction = {"action": action, "logProb": logProb, "actionReward": actionReward}

        return self.lastAction
    
    def memorize(self, 
              state: torch.Tensor, 
              action: torch.Tensor, 
              reward: torch.Tensor, 
              nextState: torch.Tensor, 
              done: int, 
              stage: dict)->dict:
        """ 
        Collect the reward and log_prob of the action.
        """

        actionReward = self.policy(nextState)
        loss = reward + self.gamma * actionReward.max() - self.lastAction['actionReward'][self.lastAction['action']]
        loss = loss*loss

        self.losses.append(loss)
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

        # train policy network
        loss = torch.stack(self.losses)
        meanLoss = torch.mean(loss)
        normLoss = (loss - meanLoss) / (torch.std(loss) + 1e-9)
        lossSum = normLoss.sum()
        self.optimizer.zero_grad()
        lossSum.backward()
        self.optimizer.step()

        self.losses = []
        self.reward = []
        self.finalReward = []

        return {'loss': meanLoss.item(), 'avgReward': avgReward, 'finalReward': finalReward}