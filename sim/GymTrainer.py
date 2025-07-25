import os
import gym
import numpy as np
import random
import torch
from IPython import display
import matplotlib.pyplot as plt
from framework.BaseAgent import BaseAgent

class GymTrainer:
    """
    The GymTrainer class is a wrapper of gym environments. It provides a simple interface to train an agent.
    """

    def __init__(self, envName: str, device='cpu', **kwargs):

        self.device = device
        self.history = {}

        self.initHyperParameters(**kwargs)
        self.initGymEnv(envName)

    #================= Gym Getter =================
    def actionSize(self)->int:
        return self.env.action_space.n

    #================= Gym Initailizer =================
    def setAgent(self, agent: BaseAgent):
        self.agent = agent

    def initHyperParameters(self, **kwargs)->None:
        keys = kwargs.keys()
        
        self.maxEpisode = kwargs['maxEpisode'] if 'maxEpisode' in keys else 500
        self.maxBatch = kwargs['maxBatch'] if 'maxBatch' in keys else 5
        self.maxStep = kwargs['maxStep'] if 'maxStep' in keys else 1000
        self.seed = kwargs['seed'] if 'seed' in keys else None

    def initGymEnv(self, envName: str)->None:

        print("=============Initializing=============")
        print(f"Initializing Gym Environments of {envName}")

        print("init envs")
        self.env = gym.make(envName, render_mode='rgb_array')

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
        self.env.np_random = np.random.default_rng(seed=self.seed) 
        self.env.action_space.seed(self.seed)

    #================= Gym Wrapper =================
    # gym warpper that transform the output of env to torch tensor
    def reset(self)->torch.Tensor:
        state = self.env.reset()[0]
        return torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=False)
    
    def step(self, action: torch.Tensor)->tuple:
        next_state, reward, done, _, _ = self.env.step(action.item())

        return torch.tensor(next_state, dtype=torch.float32, device=self.device, requires_grad=False),\
                torch.tensor((reward,), dtype=torch.float32, device=self.device, requires_grad=False),\
                done

    #================= Gym History =================
    def addHistory(self, meta: dict)->None:
        """
        Append each items in meta to the history.
        """

        for key in meta.keys():
            if key not in self.history.keys():
                self.history[key] = [meta[key]]
            else:
                self.history[key].append(meta[key])

    def getHistory(self, key: str)->list:
        """
        Get the item with key from the history.
        """

        return self.history[key]

    def removeHistory(self, key: str)->None:
        """
        Remove the item with key from the history.
        """

        del self.history[key]

    def clearHistory(self)->None:
        """
        Clear the history.
        """
        
        self.history = {}

    def plotHistory(self, clear=True, figsize=(20, 6), width=3)->None:
        """
        Plot the history of the training process.
        """

        if clear:
            display.clear_output(wait=True)

        plt.figure(figsize=figsize)

        plotCount = len(self.history.keys())
        height = plotCount//width + 1

        i=0
        for key in self.history.keys():
            i+=1
            plt.subplot(height, width, i)
            plt.plot(self.history[key])
            plt.title(key)

        plt.show()

    #================= Gym Test =================
    def test(self, agent: BaseAgent, episode=5, maxStep=5000, plot=False)->None:
        """
        Start the testing process.
        """

        # prepare the stage data
        stage={
            'batch': 0, 
            'episode': 0, 
            'step': 0,
            'maxEpisode': 0,
            'maxBatch': episode,
            'maxStep': maxStep}

        print("=============Start Testing=============")
        for ep in np.arange(self.maxBatch):
            stage['episode'] = ep
            state = self.reset()

            img = plt.imshow(self.env.render())

            for step in np.arange(maxStep):
                stage['step'] = step

                # act and learn
                action = agent.act(state, stage)['action']
                next_state, reward, done = self.step(action)
                stepMeta = agent.onTestStepDone(state, action, reward, next_state, done, stage)
                self.addHistory(stepMeta)

                img.set_data(self.env.render())
                display.clear_output(wait=True)
                display.display(plt.gcf())

                # update state
                state = next_state

                if done or step == maxStep-1:
                    break

            # trigger per episode operation
            episodeMeta = agent.onTestEpisodeDone(stage)
            self.addHistory(episodeMeta)

            # visualize
            if plot:
                self.plotHistory()

    #================= Gym Training =================
    def train(self, agent: BaseAgent=None, plot_cycle=0, eval_cycle=0, clearHist=True)->None:
        """
        Start the training process.
        """

        if clearHist:
            self.clearHistory()

        agent.train()

        # prepare the stage data
        stage={
            'batch': 0, 
            'total_batch': 0,
            'episode': 0, 
            'total_episode': 0,
            'step': 0,
            'total_step': 0,
            'maxEpisode': self.maxEpisode,
            'maxBatch': self.maxBatch,
            'maxStep': self.maxStep}

        print("=============Start Training=============")
        for bt in np.arange(self.maxEpisode):
            stage['batch'] = bt
            stage['total_batch'] += 1

            for ep in np.arange(self.maxBatch):
                stage['episode'] = ep
                stage['total_episode'] += 1
                state = self.reset()

                for step in np.arange(self.maxStep):
                    stage['step'] = step
                    stage['total_step'] += 1

                    # act and learn
                    action = agent.act(state, stage)['action']
                    next_state, reward, done = self.step(action)
                    stepMeta = agent.learn(state, action, reward, next_state, done, stage)
                    self.addHistory(stepMeta)

                    # update state
                    state = next_state

                    if done or step == self.maxStep-1:
                        break


                # trigger per episode operation
                episodeMeta = agent.onTrainEpisodeDone(stage)
                self.addHistory(episodeMeta)


                # visualize
                if plot_cycle>0 and stage['total_episode']%plot_cycle == 0:
                    self.plotHistory()

            # trigger per batch operation
            batchMeta = agent.onTrainBatchDone(stage)
            self.addHistory(batchMeta)
            