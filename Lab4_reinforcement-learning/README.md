# Reinforcement Learning Lab Assignment

---
In this assignment you will work with the [PC-Gym](https://maximilianb2.github.io/pc-gym/) framework for 
Reinforcement Learning (RL) in chemical process control. 
The **learning goal** is to understand the training metrics and how to tune the hyperparameters of an RL 
algorithm for improving the final performance of the trained policy.

## Resources
- Python 3.14 (recommended uv environment). Use the `project.toml` to install (`pip install .`) the dependencies 
  require to work on 
  this laboratory.
- [GitHub repository](https://github.com/faustoLagos/R7022_Lab-4_PCGym_2025)

## Task 1:
### Goal:
Train PPO, SAC, and DDPG policies on the [CSTR](https://maximilianb2.github.io/pc-gym/env/cstr/) environment using 
the default hyperparameters from the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) (SB3) 
implementation and compare its performance against the NMPC oracle available on PC-Gym.

The instructions are specified in the `PCGym_lab.ipynb` Jupyter notebook.

#### Evaluation:
- Visualize the performance plots. (5 pts.)
- Discuss the performance of the three algorithms on the CSTR environment. (5 pts.)

## Task 2:
### Goal:
Add realistic disturbances and constraints to approximate the training environment to a real scenario, closing the 
sim-to-real gap.

The Jupyter notebook has an environment defined including disturbances in the temperature sensor and constrains 
for the inlet temperature. In this task you must train three new policies (PPO, SAC, and DDPG) using the default 
hyperparameters for each algorithm, then after analyzing the training metrics logged in tensorboard, pick the 
algorithm with the worst performance and propose a new set of hyperparameters, train the agent again, and compare the 
final performances as in task 1.  

#### Evaluation:
- Visualize the performance plots of the algorithm with the worst performance, before and after fine-tuning the 
  hyperparameters. (5 pts.)
- Discuss the hyperparameters tuning based on the tensorboard logs. (5 pts.)

## Task 3 - after the lab session:
### Goal:
Study the impact of the reward function on the performance of the RL algorithm.

Using the environment defined in the task 1.1, train a policy with one of the studied algorithms, fine-tuning it if 
needed and compare the policy performance against the NMPC Oracle. 

#### Evaluation:
- Visualize the performance plots. (5 pts.)
- Discuss on the selected algorithm. (5 pts.)
