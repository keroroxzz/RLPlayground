import abc
import numpy as np
from typing import Tuple, Any, Callable, Optional, overload, TypeVar, Union
from typing_extensions import Self
from torch import device, dtype, Tensor
from torch._prims_common import DeviceLikeType

# torch libs
import torch
from torch.distributions import Categorical

# local libs
from utils import *

class ActionData:
    """ 
    A simple class to store the action data of the agent. 

    action: The action taken by the agent as input for gym.
    others: The data to be passed to the agent, which will be a member in ActionData

    a = ActionData(action = action, probs = probs)
    a.probs => directly get the probs data
    """
    def __init__(self, action: torch.Tensor, **kwargs: dict[str, torch.Tensor]):
        self.action = action
        for key, val in kwargs.items():
            setattr(self, key, val)

    def getAction(self) -> np.ndarray:
        """
        Get the action in the format of numpy array.
        """
        return self.action.detach().cpu().numpy()

class BaseAgent(abc.ABC, torch.nn.Module):
    """ The Base Agent Class defines the interfaces of an agent. """
    def __init__(self, action_space, device='cpu'):
        super().__init__()
        self.action_space = action_space
        self.device = device

    #===================== BaseAgent Implementations =====================#
    def to(
            self, 
            device: Optional[DeviceLikeType],
            **kwargs) -> Self:
        """
        Move the agent to a device and set the device attribute.
        """
        self.device = device
        return super().to(device=device, **kwargs)

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

    #==================== Testing Update Callbacks ======================#
    def onTestStepDone(self, 
              state: torch.Tensor, 
              action: torch.Tensor, 
              reward: torch.Tensor, 
              nextState: torch.Tensor, 
              done: int, 
              stage: Stage)->list[GraphPoint]:
        """ 
        This function will run after every taken action. 
        
        state: the current state of the environment.
        action: the action taken by the agent.
        reward: the reward received from the environment.
        nextState: the state of the environment after taking the action.
        done: whether the episode has ended.
        stage: the current stage of training or testing.

        return: a list of data points to be recorded in TensorBoard.
        """
        return []

    def onTestEpisodeDone(self, stage: Stage)->list[GraphPoint]:
        """ 
        This function will only run once every episode. 
        
        return: a list of data points to be recorded in TensorBoard.
        """
        return []
    
    #===================== abstract methods ======================#
    # The following methods are abstract methods that need to be  #
    # implemented for different Reinforcement Learning algorithms.#
    #=============================================================#
    @abc.abstractmethod
    def act(self, state: torch.Tensor, stage: Stage)-> ActionData:
        """ 
        The act method defines the agent's policy. 
        
        state: the current state of the environment.
        stage: the current stage of training or testing.
        return: the input for the gym and tensors for the agent to train itself
        """
        pass
    
    @abc.abstractmethod
    def memorize(self, 
              states: torch.Tensor, 
              actions: ActionData, 
              rewards: torch.Tensor, 
              nextStates: torch.Tensor, 
              dones: torch.Tensor, 
              stage: Stage)->list[GraphPoint]:
        """ 
        Memorize the action and return via interaction.
        This function will run after every taken action in the training stage. 
        
        state: the current state of the environment.
        action: the action taken by the agent.
        reward: the reward received from the environment.
        nextState: the state of the environment after taking the action.
        done: whether the episode has ended.
        stage: the current stage of training or testing.

        return: a list of data points to be recorded in TensorBoard.
        """
        pass

    # @abc.abstractmethod
    # def onTrainEpisodeDone(self, stage: Stage)->list[GraphPoint]:
    #     """ 
    #     Learn from the collected experience.
    #     This function will only run once every episode. 
        
    #     return: a list of data points to be recorded in TensorBoard.
    #     """
    #     pass

    @abc.abstractmethod
    def onTrainBatchDone(self, stage: Stage)->list[GraphPoint]:
        """ 
        Learn from the collected experience.
        This function will only run once every batch. 
        
        return: a list of data points to be recorded in TensorBoard.
        """
        pass