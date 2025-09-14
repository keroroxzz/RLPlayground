import numpy as np

class Stage:
    def __init__(self, maxEpisode=0, batchSize=0, maxStep=0):
        self.totalBatch = 0
        self.episode = 0
        self.totalEpisode = 0
        self.step = np.zeros(0)
        self.total_step = 0
        self.maxEpisode = maxEpisode
        self.batchSize = batchSize
        self.maxStep = maxStep