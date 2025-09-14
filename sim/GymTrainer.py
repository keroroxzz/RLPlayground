import os
import gymnasium as gym
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython import display

# torch libs
import torch
from torch.utils.tensorboard import SummaryWriter

# local libs
from utils import *
from framework.BaseAgent import BaseAgent, ActionData

writer = SummaryWriter('./runs/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

class GymTrainer:
    """
    The GymTrainer class is a wrapper of gym environments.
    It provides a simple interface to train an agent.
    """

    def __init__(self, envName: str, device='cpu', **kwargs):
        self.device = device
        self.initHyperParameters(**kwargs)
        self.initGymEnv(envName)

    #================= Gym Getter =================
    def actionSize(self)->int:
        return self.envs.single_action_space.n

    #================= Gym Initailizer =================
    def setAgent(self, agent: BaseAgent):
        self.agent = agent

    def initHyperParameters(self, **kwargs)->None:
        keys = kwargs.keys()
        
        self.maxEpisode = kwargs['maxEpisode'] if 'maxEpisode' in keys else 500
        self.batchSize = kwargs['batchSize'] if 'batchSize' in keys else 5
        self.envNum = kwargs['envNum'] if 'envNum' in keys else 6
        self.maxStep = kwargs['maxStep'] if 'maxStep' in keys else 1000
        self.maxStep = kwargs['maxStep'] if 'maxStep' in keys else 1000
        self.seed = kwargs['seed'] if 'seed' in keys else None

    def initGymEnv(self, envName: str)->None:

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
    
    def setEnvSeed(self, seed: int)->None:
        """
        Set the seed of the environment for various commom libraries.
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
    def reset(self)->torch.Tensor:
        state, info = self.envs.reset(seed=self.seed)
        return torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=False)
    
    def step(self, action:ActionData)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nextStates, rewards, terminations, truncations, info = self.envs.step(action.getAction())
        return torch.tensor(nextStates, dtype=torch.float32, device=self.device, requires_grad=False),\
               torch.tensor(rewards, dtype=torch.float32, device=self.device, requires_grad=False),\
               torch.tensor(np.bitwise_or(terminations, truncations), dtype=torch.int32, device=self.device, requires_grad=False)

    #================= Gym History =================
    def addHistory(self, pts: list[GraphPoint])->None:
        """
        Append each items in meta to the history.
        """
        for p in pts:
            p.addTo(writer)

    #================= Gym Training =================
    def train(self, agent: BaseAgent)->None:
        """
        Start the training process.
        """
        agent.train()

        # prepare the stage data
        stage = Stage(self.maxEpisode, self.batchSize, self.maxStep)

        print("=============Start Training=============")
        # epsBar = tqdm(range(self.maxEpisode))
        while True:
            # move the agent to the same device as gym
            agent_old_device = agent.device
            agent.to(self.device)

            # reset environments
            state = self.reset()

            # start training batch
            stage.episode = 0
            stage.step = np.zeros(self.envs.num_envs)
            for _ in np.arange(self.maxStep):
                stage.step += 1
                stage.total_step += self.envs.num_envs

                # act and memorize
                actions = agent.act(state, stage)
                nextStates, rewards, dones = self.step(actions)
                stepRecord = agent.memorize(state, actions, rewards, nextStates, dones, stage)
                self.addHistory(stepRecord)

                # update state
                state = nextStates

                finishedEps = dones.sum().item()
                stage.totalEpisode += finishedEps
                stage.episode += finishedEps
                stage.step *= 1-dones.numpy()

                # if a batch is finished
                if stage.episode>self.batchSize:
                    break

            stage.totalBatch += 1

            # trigger per batch operation
            agent.to(agent_old_device)
            batchRecord = agent.onTrainBatchDone(stage)
            self.addHistory(batchRecord)

            # if a batch is finished
            if stage.totalEpisode>self.maxEpisode:
                break

    #================= Gym Test =================
    def test(self, agent: BaseAgent, episode=5, maxStep=5000, plot=False)->None:
        """
        Start the testing process.
        """

        # prepare the stage data
        stage = Stage(episode, 1, maxStep)

        print("=============Start Testing=============")
        for ep in np.arange(stage.maxEpisode):
            stage.episode = ep
            stage.totalEpisode += 1
            state = self.reset()

            img = plt.imshow(self.envs.render()[0])

            for step in np.arange(stage.maxStep):
                stage.step = step
                stage.total_step += 1

                # act and memorize
                action = agent.act(state, stage)
                next_state, reward, done = self.step(action)
                stepRecord = agent.onTestStepDone(state, action, reward, next_state, done, stage)
                self.addHistory(stepRecord)

                if step % 3 == 0:
                    img.set_data(self.envs.render()[0])
                    display.clear_output(wait=True)
                    display.display(plt.gcf())

                # update state
                state = next_state

                if done or step == maxStep-1:
                    break

            # trigger per episode operation
            episodeRecord = agent.onTestEpisodeDone(stage)
            self.addHistory(episodeRecord)
            