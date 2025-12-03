import numpy as np
import math
from collections import deque

# torch libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# local libs
from utils import *
from framework.BaseAgent import BaseAgent, ActionData
from framework.ProximalPolicyOptimization import ContinuousPolicyNetwork

def init_weights(m):
    if type(m) in (nn.Linear,):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)
    
class GaussianCriticNetwork(nn.Module):
    """
    A gaussian critic network for PPO algorithm.
    It maps the state to the probability distribution of expected total rewards,
    rather than actual value.
    """

    def __init__(self, stateNum, hiddenLayers):
        super().__init__()

        dimensions = [stateNum] + hiddenLayers + [2]
        self.net = nn.Sequential()
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.net.append(nn.Linear(din, dout))
            self.net.append(nn.ReLU())
        self.net = self.net[:-1]

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(state)
        mean, std = torch.select(logits,-1,0), torch.clamp(torch.exp(torch.select(logits,-1,1)), 0.0001, 1.0)
        return mean, std
    
class ProximalPolicyOptimizationAgent(BaseAgent):
    def __init__(
            self, 
            actionNum, 
            stateNum, 
            gamma=0.99, 
            policyLR=0.001, 
            criticLR=0.001, 
            layerActor=[48, 48, 48], 
            layerCritic=[64, 64],
            memorySize=10000,
            batchSize=2000,
            trainEpoch=70,
            eps = 0.5,
            entropyBeta = 0.01,
            lamda = 0.95,
            rwShaper = lambda rwds: torch.clamp(rwds, min = -1.0),
            trainDevice:torch.device=torch.device("cuda"),
            evalDevice:torch.device=torch.device("cpu")):
        super().__init__(actionNum, trainDevice=trainDevice, evalDevice=evalDevice)

        self.lamda = lamda
        self.gamma = gamma
        self.eps = eps
        self.rwShaper = rwShaper
        self.entropyBeta = entropyBeta
        self.memorySize = memorySize
        self.batchSize = batchSize
        self.trainEpoch = trainEpoch
        self.stateNum = stateNum#+4

        # Init the policy and critic networks
        self.policy = ContinuousPolicyNetwork(self.stateNum, actionNum, layerActor).apply(init_weights)
        self.critic = GaussianCriticNetwork(self.stateNum, layerCritic).apply(init_weights)

        # Init the optimizer for our networks
        # Choose Adam to adapt to high variance loss in RL training
        self.optimizer = optim.Adam(self.policy.parameters(), lr=policyLR)
        self.optimizerCritic = optim.Adam(self.critic.parameters(), lr=criticLR)

        # memory for training
        self.states = []
        self.rewards = []
        self.dones = []
        self.finalReward = []

        self.actions = []
        self.probs = []

    def act(self, state: torch.Tensor, stage: Stage) -> ActionData:
        if self.device == self.trainDevice:
            self.eval()
        dist = self.policy(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        return ActionData(
                    action = action,
                    state = state,
                    prob = prob,
                    dist = dist)
    
    def memorize(self, 
              states: torch.Tensor, 
              actions: ActionData, 
              rewards: torch.Tensor, 
              nextStates: torch.Tensor, 
              dones: torch.Tensor, 
              stage: Stage)->list[GraphPoint]:
        if self.rwShaper:
            rewards = self.rwShaper(rewards)

        self.states.append(actions.state)
        self.probs.append(actions.prob)
        self.actions.append(actions.action)
        self.rewards.append(rewards)
        self.dones.append(1.0-dones)
        self.finalReward.extend([r for r,d in zip(rewards, dones) if d==1])
        self.nextStates = nextStates

        return [GraphPoint("Train/mean_step_reward", stage.totalStep, rewards.mean().item())]

    def onTrainBatchDone(self, stage: Stage):
        return self.learn(stage, self.nextStates)

    def learn(self, stage: Stage, nextStates: torch.Tensor) -> list[GraphPoint]:
        """
        Current diff to existing PPO implementation:
        1. no entropy penalty
        """
        # extra value for next state
        nextValues, _ = self.critic(nextStates) # [1, env num]
        nextValues = nextValues.unsqueeze(0)#.squeeze(-1)

        # move the model to training device
        if not self.device == self.trainDevice:
            self.train()

        ### 1.Post-process Memory ###
        states = torch.stack(self.states).to(self.device) # [memory size, env num, state size]
        rewards = torch.stack(self.rewards) # [memory size, env num]
        dones = torch.stack(self.dones) # [memory size, env num]
        values, _ = self.critic(states) # [memory size, env num]
        # values = values.squeeze(-1)
        actions = torch.stack(self.actions) # [memory size, env num, action size]
        probs = torch.stack(self.probs) # [memory size, env num, action size]

        ## a.Calculate advantages by GAE ##
        gae = 0
        values_cpu = torch.cat((values.to("cpu"), nextValues))
        returns = []
        with torch.no_grad():
            for i in reversed(range(len(rewards))):
                delta = (rewards[i] + self.gamma * values_cpu[i + 1] * dones[i] - values_cpu[i])
                gae = delta + self.gamma * self.lamda * dones[i] * gae
                returns.insert(0, (gae + values_cpu[i]))
        returns = torch.stack(returns).to(self.device)
        advantages = returns - values
        
        # Organize the training data
        states = states.view(-1, self.stateNum).detach()
        actions = actions.to(self.device).view(-1, self.actionSpace).detach()
        probs = probs.to(self.device).view(-1, self.actionSpace).detach()
        returns = returns.view(-1, 1).detach()
        advantages = advantages.view(-1,1).detach()

        actorLosses = []
        criticLosses = []
        for epoch in range(self.trainEpoch):
            dataset = torch.utils.data.TensorDataset(states, actions, probs, returns, advantages)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
            for states_, actions_, probs_, returns_, advantages_ in dataloader:
                ### 2.Train Policy Network ###
                dist = self.policy(states_)
                logProbs = dist.log_prob(actions_).squeeze(-1)
                
                ## c.Calculate the importance ##
                R = torch.exp(logProbs - probs_)
                clipR = torch.clamp(R, 1.0-self.eps, 1.0+self.eps)

                ## d.Actor loss ##
                entropyLoss = dist.entropy().mean() * self.entropyBeta
                actorLoss = -(torch.min(R*advantages_, clipR*advantages_)/(dist.entropy()+1.0)).mean() - entropyLoss

                ### 3.Train Critic Network ###
                mean, std = self.critic(states_)
                entropyAppx = torch.log(std)
                criticLoss = (returns_ - mean).pow(2).mean() + entropyAppx.mean() * self.entropyBeta

                ### 4.backpropagate and optimize ###
                self.optimizer.zero_grad()
                actorLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.optimizerCritic.zero_grad()
                criticLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.optimizerCritic.step()

                actorLosses.append(actorLoss.item())
                criticLosses.append(criticLoss.item())

        ### 5. Logging & Reset ###
        meanReward = rewards.mean().item()
        meanAdvantages = returns.mean().item()
        finalReward = torch.stack(self.finalReward).mean().item()

        meanActorLoss = np.mean(actorLosses)
        meanCriticLoss = np.mean(criticLosses)
        meanLoss = meanActorLoss + meanCriticLoss

        self.states = []
        self.rewards = []
        self.dones = []
        self.finalReward = []
        self.actions = []
        self.probs = []
        
        print(f"Step:{stage.totalStep} \t \
              Episode:{stage.totalEpisode} \t \
              meanLoss: {meanLoss:.2f} \t \
              LossCritic: {meanCriticLoss:.2f} \t \
              AvgAdv: {meanAdvantages:.2f} \t \
              AvgRew: {meanReward:.2f} \t \
              FinRew: {finalReward:.2f}")

        return [GraphPoint('Train/loss', stage.totalEpisode, meanLoss), 
                GraphPoint('Train/avgAdvantages', stage.totalEpisode, meanAdvantages), 
                GraphPoint('Train/avgReward', stage.totalEpisode, meanReward), 
                GraphPoint('Train/finalReward', stage.totalEpisode, finalReward), 
                GraphPoint('Train/lossCritic', stage.totalEpisode, meanCriticLoss), 
                GraphPoint('Train/lossActor', stage.totalEpisode, meanActorLoss)]