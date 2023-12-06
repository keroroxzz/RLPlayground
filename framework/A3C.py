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
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
    
class CriticNetwork(nn.Module):
    """
    A Simple Critic Network to estimate the reward of the agent given a state.
    """

    def __init__(self, stateSpace):
        super().__init__()

        self.fc1 = nn.Linear(stateSpace, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        hid = torch.relu(self.fc1(state))
        hid = torch.relu(self.fc2(hid))
        return self.fc3(hid)
    

class A3CAgent(BaseAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionSpace, stateSpace, gamma=0.99, policyLR=0.001, criticLR=0.001):
        super().__init__(actionSpace)
        
        self.gamma = gamma

        self.policy = PolicyNetwork(actionSpace)
        self.critic = CriticNetwork(stateSpace)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=policyLR)
        self.optimizerCritic = optim.SGD(self.critic.parameters(), lr=criticLR)

        self.prevStateValue = None

        self.criticLoss = []
        self.logProbs = []
        self.rewards = []
        self.advantages = []
        self.finalReward = []

    def act(self, state: torch.Tensor, stage: dict) -> torch.Tensor:
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

        nextStateValue = self.critic(nextState)

        if self.prevStateValue is not None:
            advantage = reward + self.gamma*nextStateValue - self.prevStateValue
        else:
            advantage = reward + self.gamma*nextStateValue - self.critic(state)

        self.prevStateValue = nextStateValue

        self.criticLoss.append(((advantage)**2).sum())
        self.advantages.append(advantage.item())

        self.logProbs.append(self.lastAction['logProb'])
        self.rewards.append(reward)

        if done or stage['step'] == stage['maxStep']:
            self.finalReward.append(reward)
            self.prevStateValue = None

        
        return {}

    def onTrainEpisodeDone(self, stage: dict):
        """ 
        Learn from the collected experience.
        This function will only run once every episode. 
        
        return: a dictionary of meta data {'key': value} to be recorded.
        """
        
        return {}
    
    def onTrainBatchDone(self, stage: dict):

        self.optimizerCritic.zero_grad()
        lossCritic = torch.stack(self.criticLoss).mean()
        lossCritic.backward(retain_graph=True)
        self.optimizerCritic.step()
        self.criticLoss = []

        advantages = torch.tensor(np.asarray(self.advantages), device=self.device, dtype=torch.float32)
        logProbs = torch.stack(self.logProbs)

        # train policy network
        loss = (-logProbs * advantages).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avgAdvantages = advantages.sum().item()/len(self.finalReward)
        rewards = torch.cat(self.rewards, dim=0)
        avgReward = rewards.sum().item()/len(self.finalReward)
        finalReward = torch.cat(self.finalReward, dim=0).mean().item()

        self.logProbs = []
        self.advantages = []
        self.criticRewards = []
        self.rewards = []
        self.finalReward = []

        return {'loss': loss.item(), 'avgAdvantages': avgAdvantages, 'avgReward': avgReward, 'finalReward': finalReward, 'lossCritic': lossCritic.item()}