import numpy as np
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

def init_weights(m):
    if type(m) in (nn.Linear,):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ContinuousPolicyNetwork(nn.Module):
    """
    A Continuous Policy Network outputs "multiple" probability distributions for each possible actions given a state (observation).
    The difference between the discrete policy is:
    1. The discrete policy tries to decide an action to take, while the continuous policy decide the value for 
    """
    def __init__(self, stateNum, actionNum, hiddenLayers):
        """
        stateNum: The number of variables in the state (observation)
        actionNum: The number of possible action
        layer: The number of variables in the state (observation)
        """
        super().__init__()

        self.actionNum = actionNum
        hiddenLayers = [stateNum] + hiddenLayers + [actionNum*2]
        self.net = nn.Sequential()
        for din, dout in zip(hiddenLayers[:-1], hiddenLayers[1:]):
            self.net.append(nn.Linear(din, dout))
            self.net.append(nn.ReLU())
        self.net = self.net[:-1]  # remove the last ReLU layer

    def forward(self, state) -> torch.distributions.Normal:
        logits = self.net(state)
        mean, std = torch.tanh(logits[:,:self.actionNum]), torch.clamp(torch.exp(logits[:,self.actionNum:]), 0.0001, 1.0)
        return Normal(mean, std)
    
class CriticNetwork(nn.Module):
    """
    A Critic Network estimates the expected total future rewards (value) the agent can get from a given state.
    But how does this help training?

    Consider a sequence of states and actions:
    States:  s1, s2, s3, ..., sN
    Actions: a1, a2, a3, ..., aN
    Rewards: r1, r2, r3, ..., rN

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

    def __init__(self, stateNum, hiddenLayers):
        super().__init__()

        dimensions = [stateNum] + hiddenLayers + [1]
        self.net = nn.Sequential()
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.net.append(nn.Linear(din, dout))
            self.net.append(nn.ReLU())
        self.net = self.net[:-1]  # remove the last ReLU layer

    def forward(self, state) -> torch.Tensor:
        # Mapping the state to the expected total rewards.
        # We choose ReLU as our activation function to allow unbounded output 
        # as the reward is not explicitly bounded like action probability does.
        return self.net(state)
    
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
        self.critic = CriticNetwork(self.stateNum, layerCritic).apply(init_weights)

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
        nextValues = self.critic(nextStates).unsqueeze(0).squeeze(-1) # [1, env num]

        # move the model to training device
        if not self.device == self.trainDevice:
            self.train()

        ### 1.Post-process Memory ###
        states = torch.stack(self.states).to(self.device) # [memory size, env num, state size]
        rewards = torch.stack(self.rewards) # [memory size, env num]
        dones = torch.stack(self.dones) # [memory size, env num]
        values = self.critic(states).squeeze(-1) # [memory size, env num]
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
                actorLoss = -torch.min(R*advantages_, clipR*advantages_).mean() - entropyLoss

                ### 3.Train Critic Network ###
                criticLoss = (returns_ - self.critic(states_)).pow(2).mean()

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