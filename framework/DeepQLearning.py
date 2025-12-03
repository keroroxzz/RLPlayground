import numpy as np

# torch libs
import torch
import torch.nn as nn
import torch.optim as optim

# local libs
from utils import *
from framework.BaseAgent import BaseAgent, ActionData
    
class QNetwork(nn.Module):
    """
    A Q-network is a replacement to the Q-table in classical Q-learning. 
    The objective is identical to Q-table:
        estimate the expected total future rewards (values) of possible actions at a given state.

    The advantages of Q-network in comparison to Q-table are
    1. more feasible for infinite, continuous state.
    2. the ability to generalize among unseen (but similar) states.
    
    How do we really train this network?
    Let's consider a sequence of states and actions:
    States:  s1, s2, s3, ..., sN
    Actions: a1, a2, a3, ..., aN
    Rewards: r1, r2, r3, ..., rN

    let Q(si, ai) = ri + ri+1 + ri+2 + ... + rN be the total future rewards after taking action ai at state si.

    We want to know how much reward each action could get at a state si.
    One could naively train a network to directly predict the value by input the state and action:
    Q(si, ai) = ri + ri+1 + ri+2 + ... + rN

    But there is a better way to do this.
    Q(si) = [Q(si, a1), Q(si, a2), Q(si, a3), ..., Q(si, aM)] where M is the number of possible actions.
    This allows us to evaluate the values of all possible actions at once, which is more efficient.
    """

    def __init__(self, stateNum, actionNum, layer):
        super().__init__()

        # Initialize a neural network with 2 layers. 
        self.fc1 = nn.Linear(stateNum, layer[0])
        self.fc2 = nn.Linear(layer[0], actionNum)

    def forward(self, state):
        # Mapping the state to the expected value for all actions.
        hid = torch.relu(self.fc1(state))
        return self.fc2(hid)
    

class DeepQLearning(BaseAgent):
    """
    A Deep Q-Learning Agent that uses Monte Carlo method to estimate the Q function.
    """

    def __init__(self, actionNum, stateNum, gamma=1.0, qNetLR=0.001, layerQNet=[16]):
        super().__init__(actionNum)
        
        self.gamma = gamma
        self.actionNum = actionNum

        # Init the policy and deep Q-network
        self.qnet = QNetwork(stateNum, actionNum, layerQNet)

        # Init the optimizer for our network
        # Choose Adam to adapt to high variance loss in RL training
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=qNetLR)

        # Memory for training
        self.states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.finalReward = []

    def act(self, state: torch.Tensor, stage: Stage) -> ActionData:
        """
        In the Q-learning, we don't have an explicit actor.
        But, we can take action based on the Q network!
        The most naive way is a greedy policy, which always take the action with maximum Q value at every state!
        Thus, as we find a Q-network, we also implicitly find an actor that acts the best for this Q.
        """

        # We assign a converging probability to pick random action.
        # This encourages the agent to explore more situation at the beginning of the training.
        randomActionProbability = 1.0 - stage.episode/stage.maxEpisode + 0.1 # plus 0.1 to always have at least 10% random action
        if self.device == self.trainDevice and np.random.rand() < randomActionProbability:
            # choose a random action with probability epsilon
            action = torch.randint(low=0, high=self.actionNum, size=state.shape[0:1])
        else:
            # greedily choose an action with max Q value
            values = self.qnet(state)
            action = torch.argmax(values, dim=-1)

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
        ### 1.Post-process Memory ###
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        finalReward = torch.stack(self.finalReward)

        # calculate normalized discounted rewards
        discountedRewards = rewards.clone()
        for i in range(discountedRewards.shape[0]-2, -1, -1):
            # Ri = ri + gamma * Ri+1
            discountedRewards[i] += self.gamma * discountedRewards[i+1] * (1-dones[i])

        # normalize the rewards to:
        # 1. prevent all positive or negative rewards
        # 2. stabilize the training by normalize the variance of rewards
        discountedRewards = (discountedRewards - discountedRewards.mean()) / (torch.std(discountedRewards) + 1e-9)

        # move to networks device for training
        states = states.to(self.device)
        actions = actions.to(self.device)
        discountedRewards = discountedRewards.to(self.device)

        ### 2.Training Process ###
        # train critic network
        values = self.qnet(states).squeeze(-1)

        # get the estimated values corresponding to the taken actions
        actionValues = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # calculate the loss (the difference between the estimated values and actual values)
        loss = nn.MSELoss()(actionValues, discountedRewards)

        # reset the gradient in the network
        self.optimizer.zero_grad()

        # Calculate the difference of the parameters need to decrease the loss
        loss.backward()

        # Apply the difference of the parameters
        self.optimizer.step()

        ### 3. Logging ###
        rewards = torch.cat(self.rewards, dim=0)
        averageTotalReward = rewards.sum().item()/stage.episode
        finalReward = finalReward.mean().item()

        self.states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.discounted_rewards = []
        self.finalReward = []

        print(f"Batch:{stage.totalBatch} \t Episode:{stage.totalEpisode} \t Loss: {loss.item():.2f} \t AverageTotalReward: {averageTotalReward:.2f} \t FinRew: {finalReward:.2f}")

        return [GraphPoint('Train/loss', stage.totalEpisode, loss.item()), 
                GraphPoint('Train/averageTotalReward', stage.totalEpisode, averageTotalReward), 
                GraphPoint('Train/finalReward', stage.totalEpisode, finalReward)]