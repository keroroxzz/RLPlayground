# Reinforcement Learning Playground

This is a playground to implement Reinforcement Algorithms for personal interest.

## Installation

    # for cuda 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # gym
    conda install -c conda-forge gym
    conda install -c conda-forge box2d-py
    pip install pygame

## RL Algorithms

### Done

1. PolicyGradient

### To-be Done

1. QLearning
2. DQN
3. A3C (WIP)
4. PPO

## Basic Classes

### GymTrainer

A class, GymTrainer, warps the gym simulation environment to train a given agent.

It accepts an agent implemented based on the BaseAgent.

### BaseAgent

This class defines the basic RL agent behaviors and abstract functions to be implemented for different RL algorithms.