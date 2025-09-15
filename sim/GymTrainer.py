import os
import gymnasium as gym
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from IPython import display

# torch libs
import torch
from torch.utils.tensorboard import SummaryWriter

# local libs
from utils import *
from framework.BaseAgent import BaseAgent, ActionData

class GymTrainer:
    """
    The GymTrainer class is a wrapper of gym environments.
    It provides a simple interface to train an agent.
    """

    def __init__(
            self, 
            envName: str, 
            evalDevice='cpu', 
            trainDevice='cpu', 
            **kwargs):
        """
        Create the GymTrainer to handle the training of any agent inheriting the BaseAgent class.

        envName: The Gym environment name, ex: 'LunarLander-v3'
        evalDevice: the device to put the state, reward, and other env outputs, recommend "cpu" if your network is small
        trainDevice: the device to train the network, 
        """
        self.envName = envName
        self.evalDevice = evalDevice
        self.trainDevice = trainDevice
        self.initHyperParameters(**kwargs)
        self.initGymEnv(envName)

    #================= Gym Getter =================
    def actionSize(self) -> int:
        """
        Get the number of posible actions to do.
        This is usually the output size of the policy network.
        """
        return self.envs.single_action_space.shape[0]

    #================= Gym Initailizer =================
    def initHyperParameters(
            self, 
            **kwargs) -> None:
        """
        Initialize the custom hyperparamters for the simulator.

        kwargs: The extra args from the __init__
        """
        keys = kwargs.keys()
        
        self.maxEpisode = kwargs['maxEpisode'] if 'maxEpisode' in keys else 500
        self.batchSize = kwargs['batchSize'] if 'batchSize' in keys else 5
        self.envNum = kwargs['envNum'] if 'envNum' in keys else 6
        self.maxStep = kwargs['maxStep'] if 'maxStep' in keys else 1000
        self.maxStep = kwargs['maxStep'] if 'maxStep' in keys else 1000
        self.stepLimitPenalty = kwargs['stepLimitPenalty'] if 'stepLimitPenalty' in keys else 0
        self.seed = kwargs['seed'] if 'seed' in keys else None

    def initGymEnv(
            self, 
            envName: str) -> None:
        """
        Initialize the gym simulators.

        envName: The Gym environment name, ex: 'LunarLander-v3'
        """

        print("=============Initializing=============")
        print(f"Initializing Gym Environments of {envName}")

        print("init envs")
        self.envs = gym.make_vec(
            id=envName, 
            num_envs=self.envNum, 
            render_mode='rgb_array')

        if self.seed is not None:
            print(f"set seeds {self.seed}")
            self.setEnvSeed(self.seed)
    
    def setEnvSeed(
            self, 
            seed: int) -> None:
        """
        Set the seed of the environment for various commom libraries.

        seed: The random seed to fix environment and pytorch, etc.
        """

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
        # rand lib seed
        random.seed(seed)
        np.random.seed(seed)

        # torch seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # torch env
        torch.use_deterministic_algorithms(True)

        # gym env
        # self.env.np_random = np.random.default_rng(seed=self.seed) 
        self.envs.action_space.seed(self.seed)

    #================= Gym Wrapper =================
    # gym warpper that transform the output of env to torch tensor
    def reset(self) -> torch.Tensor:
        """
        Reset the environment to a random initial state.

        action: The output from BaseAgent.act() carrying the action
        """
        state, info = self.envs.reset(seed=self.seed)
        return torch.tensor(state, dtype=torch.float32, device=self.evalDevice, requires_grad=False)
    
    def step(
            self, 
            action : ActionData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Append each items in meta to the history.

        action: The output from BaseAgent.act() carrying the action
        """
        nextStates, rewards, terminations, truncations, info = self.envs.step(action.getAction())
        return torch.tensor(nextStates, dtype=torch.float32, device=self.evalDevice, requires_grad=False),\
               torch.tensor(rewards, dtype=torch.float32, device=self.evalDevice, requires_grad=False),\
               torch.tensor(np.bitwise_or(terminations, truncations), dtype=torch.int32, device=self.evalDevice, requires_grad=False)

    #================= Gym History =================
    def addHistory(
            self, 
            pts: list[GraphPoint],
            writer: SummaryWriter|None = None) -> None:
        """
        Write to the tensorboard logging. 

        pts: list of data points to record
        """
        if writer is None:
            return
        for p in pts:
            p.addTo(writer)

    def makeSummaryWriter(
            self,
            agent:BaseAgent) -> tuple[str, SummaryWriter]:
        """
        Make a summary writer for Tensorboard logging.

        agent: The agent to be logged

        return: Name of the logging folder, SummaryWriter
        """
        name = f"{self.envName}-{agent.__class__.__name__}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}"
        return name, SummaryWriter("./runs/" + name)

    #================= Gym Training =================
    def train(
            self, 
            agent: BaseAgent,
            writer: SummaryWriter|None = None) -> None:
        """
        Start the training process.

        agent: Agent to be trained
        writer: The summary writer for Tensorboard logging, use None to disable logging
        """
        # prepare the stage data
        stage = Stage(self.maxEpisode, self.batchSize, self.maxStep)

        print("=============Start Training=============")
        while True:
            # move the agent to the same device as gym output
            agent.train() 
            agent.to(self.evalDevice)

            # reset environments
            state = self.reset()

            # start training batch
            stage.episode = 0
            stage.step = np.zeros(self.envNum)
            for s in np.arange(stage.maxStep):

                # act and memorize
                actions = agent.act(state, stage)
                nextStates, rewards, dones = self.step(actions)

                finishedEps = dones.sum().item()
                stage.totalEpisode += finishedEps
                stage.episode += finishedEps
                stage.total_step += self.envNum
                stage.step = (stage.step+1) * (1-dones.numpy())

                # handle the case that no env is finished 
                if s == stage.maxStep-1 and stage.episode == 0:
                    stage.episode = self.envNum
                    dones += 1
                    rewards -= self.stepLimitPenalty
                    print(f"Warning: No environment has finished after exceeding the maxStep {stage.maxStep}!\
                          Maybe consider setting a larger maxStep?")

                # memorize this step
                stepRecord = agent.memorize(state, actions, rewards, nextStates, dones, stage)
                self.addHistory(stepRecord, writer)
                
                # if a batch is finished
                if stage.episode>=stage.batchSize:
                    break

                # update state
                state = nextStates

            stage.totalBatch += 1

            # trigger per batch operation
            agent.to(self.trainDevice)
            batchRecord = agent.onTrainBatchDone(stage)
            self.addHistory(batchRecord, writer)

            # if a batch is finished
            if stage.totalEpisode>self.maxEpisode:
                break

    #================= Gym Test =================
    def test(
            self, 
            agent: BaseAgent, 
            episode: int = 5, 
            maxStep: int = 10000, 
            figsize: tuple[int,int] = (18, 2), 
            renderStep: int | None = None,
            writer: SummaryWriter|None = None) -> None:
        """
        Start the testing process.

        agent: Agent to be eval/tested
        episode: Number of episodes
        maxStep: The maximum step
        figSize: The figure size to render all envs
        renderStep: The step to be rendered, use None to disable rendering
        writer: The summary writer for Tensorboard logging, use None to disable logging
        """
        # set the agent to eval mode
        agent.eval()
        agent.to(self.evalDevice)

        # prepare the stage data
        stage = Stage(episode, 1, maxStep)

        print("=============Start Testing=============")
        states = self.reset()
        averageTotalReward = 0.0
        maxTotalReward = None
        minTotalReward = None
        cumulativeRewards = torch.zeros(self.envNum)

        # init plot for render
        if renderStep is not None:
            _, ax = plt.subplots(figsize=figsize)
            img = ax.imshow(np.concatenate(self.envs.render(), axis=1))

        for step in np.arange(stage.maxStep):
            stage.step = step
            stage.total_step += self.envNum

            # act and memorize
            actions = agent.act(states, stage)
            nextStates, rewards, dones = self.step(actions)
            stepRecord = agent.onTestStepDone(states, actions, rewards, nextStates, dones, stage)
            self.addHistory(stepRecord, writer)

            if renderStep is not None and step % renderStep == 0:
                img.set_data(np.concatenate(self.envs.render(), axis=1))
                display.clear_output(wait=True)
                display.display(plt.gcf())

            # update state and cumulative rewards
            states = nextStates
            cumulativeRewards = cumulativeRewards + rewards

            finishedRewards = cumulativeRewards[dones == 1]
            numFinishedEnv = finishedRewards.shape[0]
            if numFinishedEnv > 0:
                # init max min total rewards
                if maxTotalReward is None or minTotalReward is None:
                    maxTotalReward = finishedRewards[0]
                    minTotalReward = finishedRewards[0]

                # record total rewards of finished env
                averageTotalReward += finishedRewards.sum()
                maxTotalReward = max(finishedRewards.max().cpu().item(), maxTotalReward)
                minTotalReward = min(finishedRewards.min().cpu().item(), minTotalReward)

                # reset the total rewards and steps of finished envs
                cumulativeRewards[dones == 1] = 0.0
                stage.step *= 1-dones.numpy()

                # if target episode num is reached
                stage.episode += numFinishedEnv
                if stage.episode>=stage.maxEpisode:
                    break

            # trigger per episode operation
            episodeRecord = agent.onTestEpisodeDone(stage)
            self.addHistory(episodeRecord, writer)

        averageTotalReward /= stage.episode
        print(f"Average Total Reward:{averageTotalReward} \t Max Total Reward:{maxTotalReward} \t Min Total Reward:{minTotalReward}")
            