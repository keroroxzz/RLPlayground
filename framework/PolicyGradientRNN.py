
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from framework.BaseAgent import BaseAgent

class RNNPolicyNetwork(nn.Module):
    """
    A Simple Policy Network to estimate the policy of the agent given a state.
    """

    def __init__(self, actionSpace, stateSpace, hiddenLayer=16, hiddenState=16):
        super().__init__()

        self.actionSpace = actionSpace
        self.hiddenState = hiddenState
        self.fc1 = nn.Linear(stateSpace+hiddenState, hiddenLayer)
        self.fc2 = nn.Linear(hiddenLayer, hiddenLayer)
        self.fc3 = nn.Linear(hiddenLayer, actionSpace+hiddenState)

    def getInitHiddenState(self, maxBatch):
        return torch.zeros((self.hiddenState), dtype=torch.float32, requires_grad=False, device=self.fc1.weight.device)

    def forward(self, state, hiddenState=None):
        if hiddenState is None:
            hiddenState = self.getInitHiddenState(state.size(0))
        state = torch.cat([state, hiddenState], dim=-1)
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(hid)
        hid = self.fc3(hid)
        return F.softmax(hid[:self.actionSpace], dim=-1), hid[self.actionSpace:]
    

class RNNPolicyGradientAgentMC(BaseAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionSpace, gamma=0.99, lr=0.001):
        super().__init__(actionSpace)
        
        self.gamma = gamma
        self.hiddenState = None

        self.policy = RNNPolicyNetwork(actionSpace, 8, 32, 6)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)

        self.logProbs = []
        self.reward = []
        self.finalReward = []

    def act(self, state: torch.Tensor, stage: dict = None) -> torch.Tensor:
        actionProb, self.hiddenState = self.policy(state, self.hiddenState)
        actionDist = Categorical(actionProb)
        action = actionDist.sample()
        logProb = actionDist.log_prob(action)

        self.lastAction = {"action": action, "logProb": logProb, "actionProb": actionProb}

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

        self.hiddenState = None
        
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