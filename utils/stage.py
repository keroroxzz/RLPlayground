import numpy as np

class Stage:
    def __init__(
            self, 
            maxEpisode=0, 
            envNum=1,
            batchSize=None, 
            maxStep=None):
        # the number of parallel environments
        self.envNum = envNum

        # the total number of batches processed
        self.totalBatch = 0

        # the total finished episode number
        self.totalEpisode = 0

        # the total step number
        self.totalStep = 0

        # the maximum episode number to run
        self.maxEpisode = maxEpisode

        # the maximum batch size, could be None
        self.batchSize = batchSize

        # the maximum step size per episode, could be None
        self.maxStep = maxStep

        # the finished episode number in this batch
        self.episode = 0
        
        # the step number in current episode for each environment
        self.step = np.zeros(self.envNum)

        # the step number in current batch
        self.batchStep = 0

        assert self.maxEpisode is not None, "maxEpisode number is None!"
        assert self.maxEpisode > 0, "maxEpisode number must be greater than 0."
        assert self.envNum > 0, "envNum number must be greater than 0."

    def __str__(self):
        return f"totalBatch:{self.totalBatch} totalEpisode:{self.totalEpisode} totalStep:{self.totalStep} step:{self.step}"
    
    def resetStageStates(self):
        self.episode = 0
        self.batchStep = 0
        self.step = np.zeros(self.envNum)
        
    def updatePerStepStates(self, dones: np.ndarray):
        finishEnvs = dones.sum()
        self.totalEpisode += finishEnvs
        self.episode += finishEnvs
        self.totalStep += self.envNum
        self.step = (self.step+1) * (1-dones)
        self.batchStep += 1

    def updatePerBatchStates(self):
        self.totalBatch += 1

    def isMaxStepReached(self):
        return self.maxStep is not None and self.step.max() >= self.maxStep
    
    def isBatchSizeReached(self):
        return self.batchSize is not None and self.batchStep >= self.batchSize
    
    def isMaxEpisodeReached(self):
        return self.maxEpisode is not None and self.totalEpisode >= self.maxEpisode