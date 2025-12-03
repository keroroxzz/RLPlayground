import numpy as np

# torch libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

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
    def __init__(self, stateNum, actionNum, intermediateDimensions):
        """
        stateNum: The number of variables in the state (observation)
        actionNum: The number of possible action
        layer: The number of variables in the state (observation)
        """
        super().__init__()

        intermediateDimensions = [stateNum] + intermediateDimensions + [actionNum]
        self.net = nn.Sequential()
        for din, dout in zip(intermediateDimensions[:-1], intermediateDimensions[1:]):
            self.net.append(nn.Linear(din, dout))
            self.net.append(nn.ReLU())
        self.net = self.net[:-1]  # remove the last ReLU layer

    def forward(self, state):
        # Mapping the state to logits, i.e, the "score" for each action.
        logits = self.net(state)

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

    def __init__(self, stateNum, intermediateDimensions):
        super().__init__()

        # Initialize a 3 layers network
        dimensions = [stateNum] + intermediateDimensions + [1]
        self.net = nn.Sequential()
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.net.append(nn.Linear(din, dout))
            self.net.append(nn.ReLU())
        self.net = self.net[:-1]  # remove the last ReLU layer

    def forward(self, state):
        # Mapping the state to the expected total rewards.
        # We choose ReLU as our activation function to allow unbounded output 
        # as the reward is not explicitly bounded like action probability does.
        return self.net(state)
    

class AdvantageActorCriticAgent(BaseAgent):
    """
    A Advantage Actor Critic Agent utilizes both a policy network and a critic network to learn the policy.
    The policy network (actor) is responsible for selecting actions based on the current state, 
    while the critic network evaluates the expected reward of a state made by the current policy.
    """

    def __init__(
            self, 
            actionNum, 
            stateNum, 
            gamma=0.99, 
            policyLR=0.001, 
            criticLR=0.001, 
            layerActor=[8, 8], 
            layerCritic=[16, 16]):
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


        return [GraphPoint("Train/mean_step_reward", stage.totalStep, rewards.mean().item())]

    def onTrainBatchDone(self, stage: Stage):
        """ 1. post-process memory """
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        finalReward = torch.stack(self.finalReward)

        # calculate normalized discounted rewards
        # why normalize the rewards?
        #  1. prevent all positive or negative rewards
        #  2. stablize the training by normalize the variance
        discountedRewards = rewards
        for i in range(discountedRewards.shape[0]-2, -1, -1):
            # Ri = ri + gamma * Ri+1
            discountedRewards[i] += self.gamma * discountedRewards[i+1] * (1-dones[i])
        # normalize the discounted rewards
        discountedRewards = (discountedRewards - discountedRewards.mean()) / (torch.std(discountedRewards) + 1e-9)

        # move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        discountedRewards = discountedRewards.to(self.device)

        """ 2.train critic network """
        # We train the critic before the policy network because:
        # 1. The policy update relies on accurate advantage, which is calculated using the critic's value estimates.
        # 2. Thus, we need to ensure the critic is well-trained base on the current policy to provide reliable feedback.
        stateValues = self.critic(states).squeeze(-1)
        lossCritic = nn.MSELoss()(stateValues, discountedRewards)
        self.optimizerCritic.zero_grad()
        lossCritic.backward()
        self.optimizerCritic.step()

        """ 3.train policy network """
        # After updating the critic, we can now compute the advantages.
        # The advantage function measures how much better (or worse) an action is compared to the average future rewards this policy can get.
        # where the discountedRewards is Q(s, a), the actuall future rewards the agent can get from a state,
        # while self.critic(states) is the expected future rewards the agent can get from a state
        # discountedRewards = Q(s, a)
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

        print(f"Batch:{stage.totalBatch} \t Episode:{stage.totalEpisode} \t Loss: {loss.item():.2f} \t AvgAdv: {avgAdvantages:.2f} \t AvgRew: {avgReward:.2f} \t FinRew: {finalReward:.2f} \t LossCritic: {lossCritic.item():.2f}")

        return [GraphPoint('Train/loss', stage.totalEpisode, loss.item()), 
                GraphPoint('Train/avgAdvantages', stage.totalEpisode, avgAdvantages), 
                GraphPoint('Train/avgReward', stage.totalEpisode, avgReward), 
                GraphPoint('Train/finalReward', stage.totalEpisode, finalReward), 
                GraphPoint('Train/lossCritic', stage.totalEpisode, lossCritic.item()), 
                GraphPoint('Train/lossActor', stage.totalEpisode, loss.item())]
    

class ContinuousAdvantageActorCriticAgent(AdvantageActorCriticAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionNum, stateNum, gamma=0.99, policyLR=0.001, criticLR=0.001, layerActor=[48, 48, 48], layerCritic=[64, 64]):
        super().__init__(actionNum*2, stateNum+4, gamma, policyLR, criticLR, layerActor, layerCritic)
        self.actionNum = actionNum

    def act(self, state: torch.Tensor, stage: Stage) -> ActionData:
        wave = torch.stack( 
            [torch.sin(torch.from_numpy(stage.step)/10.0),
            torch.sin(torch.from_numpy(stage.step)/100.0),
            torch.cos(torch.from_numpy(stage.step)/10.0),
            torch.cos(torch.from_numpy(stage.step)/100.0)], dim=-1).to(torch.float32)
        state = torch.cat([state, wave], dim=-1)
        policyOut = self.policy(state)
        actionMean = policyOut[:, :self.actionNum]
        actionStd = policyOut[:, self.actionNum:]

        # sample an action from the probability distribution over possible actinos
        actionDist = Normal(actionMean, actionStd)
        action = actionDist.sample()

        return ActionData(
                    action = action, 
                    state = state)
    
    def memorize(self, 
              states: torch.Tensor, 
              actions: ActionData, 
              rewards: torch.Tensor, 
              nextStates: torch.Tensor, 
              dones: torch.Tensor, 
              stage: Stage)->list[GraphPoint]:
        self.states.append(actions.state)
        self.actions.append(actions.action)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.finalReward.extend([r for r,d in zip(rewards, dones) if d==1])

        return [GraphPoint("Train/mean_step_reward", stage.totalStep, rewards.mean().item())]

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
            # Ri = ri + gamma * Ri+1
            discountedRewards[i] += self.gamma * discountedRewards[i+1] * (1-dones[i])
        # normalize the rewards to
        # 1. prevent all positive or negative rewards
        # 2. stablize the training by normalize the variance
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

        policyOut = self.policy(states)
        actionMean = policyOut[:, :, :self.actionNum]
        actionStd = policyOut[:, :, self.actionNum:]
        actionDist = Normal(actionMean, actionStd)

        logProbs = actionDist.log_prob(actions).squeeze(-1)
        loss = (-logProbs * advantages.unsqueeze(-1).detach()).mean()
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

        print(f"Batch:{stage.totalBatch} \t Episode:{stage.totalEpisode} \t Loss: {loss.item():.2f} \t AvgAdv: {avgAdvantages:.2f} \t AvgRew: {avgReward:.2f} \t FinRew: {finalReward:.2f} \t LossCritic: {lossCritic.item():.2f}")

        return [GraphPoint('Train/loss', stage.totalEpisode, loss.item()), 
                GraphPoint('Train/avgAdvantages', stage.totalEpisode, avgAdvantages), 
                GraphPoint('Train/avgReward', stage.totalEpisode, avgReward), 
                GraphPoint('Train/finalReward', stage.totalEpisode, finalReward), 
                GraphPoint('Train/lossCritic', stage.totalEpisode, lossCritic.item()), 
                GraphPoint('Train/lossActor', stage.totalEpisode, loss.item())]
    


class ContinuousImitationAdvantageActorCriticAgent(ContinuousAdvantageActorCriticAgent):
    """
    A Policy Gradient Agent that uses Monte Carlo method to estimate the policy gradient.
    """

    def __init__(self, actionNum, stateNum, gamma=0.99, policyLR=0.001, criticLR=0.001, layerActor=[48, 48, 48], layerCritic=[64, 64]):
        super().__init__(actionNum*2, stateNum+4, gamma, policyLR, criticLR, layerActor, layerCritic)
        self.actionNum = actionNum

    def act(self, state: torch.Tensor, stage: Stage) -> ActionData:
        wave = torch.stack( 
            [torch.sin(torch.from_numpy(stage.step)/10.0),
            torch.sin(torch.from_numpy(stage.step)/100.0),
            torch.cos(torch.from_numpy(stage.step)/10.0),
            torch.cos(torch.from_numpy(stage.step)/100.0)], dim=-1).to(torch.float32)
        state = torch.cat([state, wave], dim=-1)
        policyOut = self.policy(state)
        actionMean = policyOut[:, :self.actionNum]
        actionStd = policyOut[:, self.actionNum:]

        # sample an action from the probability distribution over possible actinos
        actionDist = Normal(actionMean, actionStd)
        action = actionDist.sample()

        return ActionData(
                    action = action, 
                    state = state)
    
    def memorize(self, 
              states: torch.Tensor, 
              actions: ActionData, 
              rewards: torch.Tensor, 
              nextStates: torch.Tensor, 
              dones: torch.Tensor, 
              stage: Stage)->list[GraphPoint]:
        self.states.append(actions.state)
        self.actions.append(actions.action)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.finalReward.extend([r for r,d in zip(rewards, dones) if d==1])

        return [GraphPoint("Train/mean_step_reward", stage.totalStep, rewards.mean().item())]

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
            # Ri = ri + gamma * Ri+1
            discountedRewards[i] += self.gamma * discountedRewards[i+1] * (1-dones[i])
        # normalize the rewards to
        # 1. prevent all positive or negative rewards
        # 2. stablize the training by normalize the variance
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizerCritic.step()

        # train policy network
        stateValues = self.critic(states).squeeze(-1)
        advantages = discountedRewards - stateValues
        avgAdvantages = advantages.sum().item()/stage.episode

        policyOut = self.policy(states)
        actionMean = policyOut[:, :, :self.actionNum]
        actionStd = policyOut[:, :, self.actionNum:]
        actionDist = Normal(actionMean, actionStd)

        logProbs = actionDist.log_prob(actions).squeeze(-1)
        loss = (-logProbs * advantages.unsqueeze(-1).detach()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
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

        print(f"Batch:{stage.totalBatch} \t Episode:{stage.totalEpisode} \t Loss: {loss.item():.2f} \t AvgAdv: {avgAdvantages:.2f} \t AvgRew: {avgReward:.2f} \t FinRew: {finalReward:.2f} \t LossCritic: {lossCritic.item():.2f}")

        return [GraphPoint('Train/loss', stage.totalEpisode, loss.item()), 
                GraphPoint('Train/avgAdvantages', stage.totalEpisode, avgAdvantages), 
                GraphPoint('Train/avgReward', stage.totalEpisode, avgReward), 
                GraphPoint('Train/finalReward', stage.totalEpisode, finalReward), 
                GraphPoint('Train/lossCritic', stage.totalEpisode, lossCritic.item()), 
                GraphPoint('Train/lossActor', stage.totalEpisode, loss.item())]