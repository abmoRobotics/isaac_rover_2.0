import threading
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env

# Omniverse Isaac Sim tutorial: Creating New RL Environment 
# https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_new_rl_example.html

# Instance of VecEnvBase and create the task
# from omni.isaac.gym.vec_env import VecEnvBase
# env = VecEnvBase(headless=True)

# from tasks.rover import RoverTask
# task = RoverTask(name="Rover")
env = load_omniverse_isaacgym_env(task_name="Rover")
env = wrap_env(env)


import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from learning.model import StochasticActorHeightmap, DeterministicHeightmap

# Define the models (stochastic and deterministic models) for the agent using mixins.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy

# Load and wrap the environment
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=60, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {  "policy": StochasticActorHeightmap(env.observation_space, env.action_space, network_features=[256,160,128], encoder_features=[80,40], activation_function="leakyrelu"),
                "value": DeterministicHeightmap(env.observation_space, env.action_space, network_features=[256,160,128], encoder_features=[80,40] ,activation_function="leakyrelu")}
# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ppo.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.05)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 60 #20
cfg_ppo["learning_epochs"] = 4 #5
cfg_ppo["mini_batches"] = 60 #5
cfg_ppo["discount_factor"] = 0.99 # 0.999
cfg_ppo["lambda"] = 0.95 # 0.99
cfg_ppo["policy_learning_rate"] = 0.0003
cfg_ppo["value_learning_rate"] = 0.0003
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0.008

agent = PPO(models=models_ppo,
            memory=memory, 
            cfg=cfg_ppo, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# # start training
# trainer.train()


# start training in a separate thread

threading.Thread(target=trainer.train).start()


# run the simulation in the main thread

env.run()