import numpy as np

# torch libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# local libs
from utils import *
from framework.BaseAgent import BaseAgent, ActionData

class PolicyNetwork(nn.Module):
    """
    A Policy Network outputs a probability distribution over possible actions given a state (observation).
    It maps a state to the action probabilities by non-linear transformations, which effectively determins what action should be taken under a certain situation. 
    For example, if an agent is approaching a wall, the policy network should assign a higher probability to the action, "decelerate", rather than "accelerate". 

    ex: State:  [0.2, 5.0], distances to the wall ahead and behind, respectively
        Output: [0.1, 0.9], 10% probability to accelerate, 90% probability to decelerate
    """
    def __init__(self, stateNum, actionNum, layerSize):
        """
        stateNum: The number of variables in the state (observation)
        actionNum: The number of possible action
        layer: The number of variables in the state (observation)
        """
        super().__init__()

        # Initialize a 3 layers network
        self.fc1 = nn.Linear(stateNum, layerSize[0])
        self.fc2 = nn.Linear(layerSize[0], layerSize[1])
        self.fc3 = nn.Linear(layerSize[1], actionNum)

    def forward(self, state):
        # Mapping the state to logits, i.e, the "score" for each action.
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        logits = self.fc3(hid)

        # Finally, use the softmax function to convert scores (logits) into probabilities.
        # Softmax(logits) = exp(logits) / sum(exp(logits))
        # The softmax output has two important properties:
        # 1. The sum of all probabilities equals 1.
        # 2. Each probability lies in the range [0, 1].
        # These properties make the output a valid probability distribution.
        return F.softmax(logits, dim=-1)
    
class CriticNetwork(nn.Module):
    """
    A Critic Network estimates the expected total future rewards (value) the agent can get from a given state.
    But how does this help training?

    Consider a sequence of states and actions:
    States:  S = s1, s2, s3, ..., sN
    Actions: A = a1, a2, a3, ..., aN
    Rewards: R = r1, r2, r3, ..., rN

    We want to know how much reward the agent can expect after observing a state si.
    One could naively define:
    Critic(si) = ri + ri+1 + ri+2 + ... + rN
    But this is not really an "average," since the future depends on the actions the agent might take.

    In reality, at state s1, the agent can take different actions, leading to different future rewards.
    For example, if Policy(s1) = [0.2, 0.8]:
    - Action 1 (probability 0.2) might lead to a reward of -100.
    - Action 2 (probability 0.8) might lead to a reward of 300.
    So the critic learns from these sampled outcomes to estimate the expected value.

    With enough training data, the critic converges to the "average" future reward:
    Critic(s1) = 0.2 * -100 + 0.8 * 300 = 220

    This allows us to evaluate how good a chosen action is relative to the expected value at that state.
    """

    def __init__(self, stateNum, layer):
        super().__init__()

        # Initialize a 3 layers network
        self.fc1 = nn.Linear(stateNum, layer[0])
        self.fc2 = nn.Linear(layer[0], layer[1])
        self.fc3 = nn.Linear(layer[1], 1)

    def forward(self, state):
        # Mapping the state to the expected total rewards.
        # We choose ReLU as our activation function to allow unbounded output 
        # as the reward is not explicitly bounded like action probability does.
        hid = torch.relu(self.fc1(state))
        hid = torch.relu(self.fc2(hid))
        return self.fc3(hid)
    

class AdvantageActorCriticAgent(BaseAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionNum, stateNum, gamma=0.99, policyLR=0.001, criticLR=0.001, layerActor=[8, 8], layerCritic=[16, 16]):
        super().__init__(actionNum)
        
        self.gamma = gamma

        # Init the policy and critic networks
        self.policy = PolicyNetwork(stateNum, actionNum, layerActor)
        self.critic = CriticNetwork(stateNum, layerCritic)

        # Init the optimizer for our networks
        # Choose Adam to adapt to high variance loss in RL training
        self.optimizer = optim.Adam(self.policy.parameters(), lr=policyLR)
        self.optimizerCritic = optim.Adam(self.critic.parameters(), lr=criticLR)

        # memory for training
        self.states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.finalReward = []

    def act(self, state: torch.Tensor, stage: Stage) -> ActionData:
        logProb = self.policy(state)

        # sample an action from the probability distribution over possible actinos
        actionDist = Categorical(logProb)
        action = actionDist.sample()

        return ActionData(action = action)
    
    def memorize(self, 
              states: torch.Tensor, 
              actions: ActionData, 
              rewards: torch.Tensor, 
              nextStates: torch.Tensor, 
              dones: torch.Tensor, 
              stage: Stage)->list[GraphPoint]:
        self.states.append(states)
        self.actions.append(actions.action)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.finalReward.extend([r for r,d in zip(rewards, dones) if d==1])

        return [GraphPoint("Train/mean_step_reward", stage.total_step, rewards.mean().item())]

    def onTrainBatchDone(self, stage: Stage):
        # post-process memory
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        finalReward = torch.stack(self.finalReward)

        # calculate normalized discounted rewards
        discountedRewards = rewards
        for i in range(discountedRewards.shape[0]-2, -1, -1):
            discountedRewards[i] += self.gamma * discountedRewards[i+1] * (1-dones[i])
        discountedRewards = (discountedRewards - discountedRewards.mean()) / (torch.std(discountedRewards) + 1e-9)

        # move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        discountedRewards = discountedRewards.to(self.device)

        # train critic network
        stateValues = self.critic(states).squeeze(-1)
        lossCritic = nn.MSELoss()(stateValues, discountedRewards)
        self.optimizerCritic.zero_grad()
        lossCritic.backward()
        self.optimizerCritic.step()

        # train policy network
        stateValues = self.critic(states).squeeze(-1)
        advantages = discountedRewards - stateValues
        avgAdvantages = advantages.sum().item()/stage.episode
        logProbs = self.policy(states).gather(-1, actions.unsqueeze(-1)).squeeze(-1).log()
        loss = (-logProbs * advantages.detach()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        rewards = torch.cat(self.rewards, dim=0)
        avgReward = rewards.sum().item()/stage.episode
        finalReward = finalReward.mean().item()

        self.states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.discounted_rewards = []
        self.finalReward = []

        print(f"Episode:{stage.totalEpisode} \t Loss: {loss.item():.2f} \t AvgAdv: {avgAdvantages:.2f} \t AvgRew: {avgReward:.2f} \t FinRew: {finalReward:.2f} \t LossCritic: {lossCritic.item():.2f}")

        return [GraphPoint('Train/loss', stage.totalEpisode, loss.item()), 
                GraphPoint('Train/avgAdvantages', stage.totalEpisode, avgAdvantages), 
                GraphPoint('Train/avgReward', stage.totalEpisode, avgReward), 
                GraphPoint('Train/finalReward', stage.totalEpisode, finalReward), 
                GraphPoint('Train/lossCritic', stage.totalEpisode, lossCritic.item()), 
                GraphPoint('Train/lossActor', stage.totalEpisode, loss.item())]