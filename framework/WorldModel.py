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
    
class WorldNetwork(nn.Module):
    """
    A World Model Network to learn the reward of the agent given a state and action.
    It enable the backpropagation through the (modeled) environment to directly train the agent.
    """

    def __init__(self, stateSpace):
        super().__init__()

        self.fc1 = nn.Linear(stateSpace, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat([state, action], dim=-1)
        hid = torch.relu(self.fc1(state))
        hid = torch.relu(self.fc2(hid))
        return self.fc3(hid)
    

class WorldModelAgent(BaseAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionSpace, stateSpace, gamma=0.99, policyLR=0.001, worldModelLR=0.001):
        super().__init__(actionSpace)
        
        self.gamma = gamma

        self.policy = PolicyNetwork(actionSpace)
        self.worldModel = WorldNetwork(stateSpace+actionSpace)
        self.optimizerPolicy = optim.SGD(self.policy.parameters(), lr=policyLR)
        self.optimizerWorldModel = optim.SGD(self.worldModel.parameters(), lr=worldModelLR)

        self.worldModelRewardInv = []
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

        # calculate advantage, that is the difference between the reward by taking the current action and the expected reward.
        nextActionProb = self.policy(nextState)
        nextStateValue = self.worldModel(nextState, nextActionProb.detach())
        advantage = reward + self.gamma*nextStateValue - self.worldModel(state, self.lastAction['actionProb'].detach())

        # calculate the reward by the world model, i.e., the imagaination of the agent.
        self.worldModelRewardInv.append(-self.worldModel(state, self.lastAction['actionProb']))

        # self.worldModelLoss.append(((advantage)**2).sum())
        self.advantages.append(advantage)

        self.logProbs.append(self.lastAction['logProb'])
        self.rewards.append(reward)

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

        # prepare advantages
        advantages = torch.stack(self.advantages).squeeze(1)
        avgAdvantages = advantages.sum().item()/len(self.finalReward)

        # world model loss
        lossWorldModel = torch.square(advantages).mean()
        self.optimizerWorldModel.zero_grad()
        lossWorldModel.backward(retain_graph=True)

        # policy loss by world model
        worldMdoelRewardLoss = torch.stack(self.worldModelRewardInv).mean()
        worldMdoelRewardLoss = worldMdoelRewardLoss / (lossWorldModel.item()+1.0) # scale the weight of the world model reward loss
        
        # policy loss by agent action sampling
        logProbs = torch.stack(self.logProbs)
        advantages = advantages / (torch.std(advantages) + 1e-9)
        sampleLoss = (-logProbs * advantages.detach()).sum()

        # train policy network
        loss = sampleLoss + worldMdoelRewardLoss 
        self.optimizerPolicy.zero_grad()
        loss.backward()

        # train world model
        self.optimizerWorldModel.zero_grad()
        lossWorldModel.backward(retain_graph=True)

        # update parameters
        self.optimizerPolicy.step()
        self.optimizerWorldModel.step()

        rewards = torch.cat(self.rewards, dim=0)
        avgReward = rewards.sum().item()/len(self.finalReward)
        finalReward = torch.cat(self.finalReward, dim=0).mean().item()

        self.worldModelRewardInv = []
        self.worldModelLoss = []
        self.criticRewards = []
        self.rewards = []
        self.logProbs = []
        self.advantages = []
        self.finalReward = []

        return {'loss': loss.item(), 'avgAdvantages': avgAdvantages, 'avgReward': avgReward, 'finalReward': finalReward, 'lossWorldModel': lossWorldModel.item()}