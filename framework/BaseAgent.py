import abc
import numpy as np
from typing import Tuple

import torch
from torch.distributions import Categorical


class BaseAgent(abc.ABC, torch.nn.Module):
    """ The Base Agent Class defines the interfaces of an agent. """

    def __init__(self, action_space, device='cpu'):
        super().__init__()
        self.action_space = action_space
        self.device = device

    #===================== BaseAgent Implementations =====================#
    def to(self, device):
        """
        Move the agent to a device and set the device attribute.
        """
        super().to(device)
        self.device = device
        return self
    
    def estimatePolicy(self, state: torch.Tensor)->torch.Tensor:
        """
        Estimate the policy of the agent by the policy module.
        """

        return self.policy(state)
    
    def sample(self, action_prob: torch.Tensor)-> Tuple[torch.Tensor]:
        """
        Sample an action and return the log probability of the action.
        """

        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action, log_prob

    def act(self, state: torch.Tensor, stage: dict = None)->torch.Tensor:
        """ 
        The act method defines the agent's policy. 
        
        state: the current state of the environment.
        stage:{
            'batch': bt, 
            'episode': ep, 
            'step': state,
            'maxEpisode': self.maxEpisode,
            'maxBatch': self.maxBatch,
            'maxStep': self.maxStep}
        return: the action to take.
        """

        self.last_action = {}

        action_prob = self.estimatePolicy(state)
        action, log_prob = self.sample(action_prob)

        self.last_action['action_prob'] = action_prob
        self.last_action['action'] = action
        self.last_action['log_prob'] = log_prob

        return self.last_action
    
    #Save & Load functions
    def save(self, path: str):
        """
        Save the state dict of the agent to a file.
        """

        torch.save(self.state_dict(), path)
        return self
    
    def load(self, path: str):
        """
        Load the state dict from a file.
        """

        self.load_state_dict(torch.load(path))
        return self

    #===================== class methods =====================#
    @classmethod
    def getActionNP(cls, action: torch.Tensor)->np.ndarray:
        """ 
        Convert a tensor action to a numpy array. 

        action: the output of the agent's act method.
        return: the corresponding numpy array.
        """
        return action.detach().cpu().numpy()

    #==================== Testing Update Callbacks ======================#
    def onTestStepDone(self, 
              state: 
              torch.Tensor, 
              action: torch.Tensor, 
              reward: torch.Tensor, 
              nextState: torch.Tensor, 
              done: int, 
              stage: dict = None)->dict:
        """ 
        This function will run after every taken action. 
        
        state: the current state of the environment.
        action: the action taken by the agent.
        reward: the reward received from the environment.
        nextState: the state of the environment after taking the action.
        done: whether the episode has ended.
        stage:{
            'episode': ep, 
            'step': state,
            'maxBatch': self.maxBatch,
            'maxStep': self.maxStep}
        """
        return {}

    def onTestEpisodeDone(self, stage: dict = None)->dict:
        """ 
        This function will only run once every episode. 
        
        return: a dictionary of meta data {'key': value} to be recorded.
        """
        return {}
    
    #===================== abstract methods ======================#
    # The following methods are abstract methods that need to be  #
    # implemented for different Reinforcement Learning algorithms.#
    #=============================================================#
    @abc.abstractmethod
    def learn(self, 
              state: torch.Tensor, 
              action: torch.Tensor, 
              reward: torch.Tensor, 
              nextState: torch.Tensor, 
              done: int, 
              stage: dict = None)->dict:
        """ 
        Collect (and Learn) from the current state.
        This function will run after every taken action. 
        
        state: the current state of the environment.
        action: the action taken by the agent.
        reward: the reward received from the environment.
        nextState: the state of the environment after taking the action.
        done: whether the episode has ended.
        stage:{
            'batch': the current batch number, 
            'episode': the current episode number within a batch, 
            'step': the current step number within a batch, 
            'maxEpisode': self.maxEpisode,
            'maxBatch': self.maxBatch,
            'maxStep': self.maxStep}

        return: a dictionary of meta data {'plot name': value} to be recorded.
        """
        pass

    @abc.abstractmethod
    def onTrainEpisodeDone(self, stage: dict = None)->dict:
        """ 
        Learn from the collected experience.
        This function will only run once every episode. 
        
        return: a dictionary of meta data {'plot name': value} to be recorded.
        """
        pass

    @abc.abstractmethod
    def onTrainBatchDone(self, stage: dict = None)->dict:
        """ 
        Learn from the collected experience.
        This function will only run once every batch. 
        
        return: a dictionary of meta data {'plot name': value} to be recorded.
        """
        pass