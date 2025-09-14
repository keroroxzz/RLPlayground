# Reinforcement Learning Playground

This is a playground to learn and implement Reinforcement Algorithms.

## Installation

1. Create conda environment (Recommend)

        conda create -n rlpg python==3.12


2. Install dependency for jupyter notebook

        pip install ipykernel matplotlib pyyaml typeguard

3. Install gymnasium

        pip install swig
        pip install gymnasium[box2d]

4. Install Pytorch

    please refer to the official [Pytorch Installation Guide](https://pytorch.org/get-started/locally/)

5. Install Tesorboad

        pip install tensorboard

### Trouble Shooting

1. "Box2D is not installed"

    First, verify the installation of Box2D.

        python3 -c "import Box2D"

    Box2D is failed to import due to mismatch version of libstdc++

        ImportError: /home/rtu/miniconda3/envs/rlpg/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found

    Try updating your libstdc++

        conda install -c conda-forge libgcc=15.1.0

## RL Algorithms

### Done
1. [PolicyGradient](https://github.com/keroroxzz/RLPlayground/blob/main/framework/PolicyGradient.py)
2. [Advantage Actro-Critic (A2C)](https://github.com/keroroxzz/RLPlayground/blob/main/framework/A2C.py)

### To-do
1. QLearning
2. [WIP] DeepQLearning (DQN)
4. ProximalPolicyOptimization (PPO)

## How to train your agent?

### 

## Basic Classes

### 1. GymTrainer [(link)](https://github.com/keroroxzz/RLPlayground/blob/main/sim/GymTrainer.py)

- A class, GymTrainer, warps the gym simulation environment to train/evaluate a given agent.

- It accepts an agent inheriting the BaseAgent class.

### 2. BaseAgent [(link)](https://github.com/keroroxzz/RLPlayground/blob/main/framework/BaseAgent.py)

- This class defines the basic RL agent behaviors and abstract functions to be implemented for different RL algorithms.

## Reference

1. Hung-yi Lee, "DRL Lecture," https://www.youtube.com/watch?v=z95ZYgPgXOY&list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_&ab_channel=Hung-yiLee, 2018.