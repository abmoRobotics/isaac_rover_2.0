import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env
from skrl.utils import set_seed
from learning.model import StochasticActorHeightmap, DeterministicHeightmap
import hydra
from omegaconf import DictConfig
from hydra import compose, initialize
import wandb
import datetime
#cfg_ppo = PPO_DEFAULT_CONFIG.copy()

# set the seed for reproducibility
set_seed(42)

#@hydra.main(config_name="config", config_path="cfg")
def parse_hydra_configs():
    initialize(config_path="cfg", job_name="test_app")
    cfg = compose(config_name="config")

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_network = cfg.trainSKRL.network
    cfg_experiment = cfg.trainSKRL.experiment
    # Set all parameters according to cfg file
    for param, value in (cfg.trainSKRL.config).items():
        cfg_ppo[param] = value
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    run_simulation(cfg_ppo, cfg_network, cfg_experiment)



def run_simulation(cfg_ppo, cfg_network, cfg_experiment):

    #print(cfg_ppo)
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    print(cfg_network.mlp.layers)
    exit()
    # Load and wrap the Omniverse Isaac Gym environment
    env = load_omniverse_isaacgym_env(task_name="Rover")
    env = wrap_env(env)
    
    device = env.device

    # Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=60, num_envs=env.num_envs, device=device)

    # Get values from cfg
    mlp_layers = cfg_network.mlp.layers
    encoder_layers = cfg_network.encoder.layers
    activation_function = cfg_network.mlp.activation

    # Instantiate the agent's models (function approximators).
    models_ppo = {  "policy": StochasticActorHeightmap(env.observation_space, env.action_space, network_features=mlp_layers, encoder_features=encoder_layers, activation_function=activation_function),
                "value": DeterministicHeightmap(env.observation_space, env.action_space, network_features=mlp_layers, encoder_features=encoder_layers ,activation_function=activation_function)}

    # Instantiate parameters of the model
    for model in models_ppo.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.05)

    # Define agent
    agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1600, "headless": False}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()

class TrainerSKRL():
    def __init__(self):
        self._load_cfg()
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb_group =f"Test-Anton_{time_str}"
        self.wandb_name = f"{time_str}"
       # self.start_simulation()
        #self.start_training()



    def _load_cfg(self):
        initialize(config_path="cfg", job_name="test_app")
        cfg = compose(config_name="config")

        self.cfg_ppo = PPO_DEFAULT_CONFIG.copy()
        self.cfg_network = cfg.trainSKRL.network
        self.cfg_experiment = cfg.trainSKRL.experiment
        # Set all parameters according to cfg file
        for param, value in (cfg.trainSKRL.config).items():
            self.cfg_ppo[param] = value
        
        print(self.cfg_ppo)
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def start_simulation(self):
        env = load_omniverse_isaacgym_env(task_name="Rover")
        self.env = wrap_env(env)

    def train(self):
        env = self.env
        device = env.device

        
        # Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
        memory = RandomMemory(memory_size=60, num_envs=self.env.num_envs, device=device)

        # Get values from cfg
        mlp_layers = self.cfg_network.mlp.layers
        encoder_layers = self.cfg_network.encoder.layers
        activation_function = self.cfg_network.mlp.activation

        # Instantiate the agent's models (function approximators).
        models_ppo = {  "policy": StochasticActorHeightmap(env.observation_space, env.action_space, network_features=mlp_layers, encoder_features=encoder_layers, activation_function=activation_function),
                    "value": DeterministicHeightmap(env.observation_space, env.action_space, network_features=mlp_layers, encoder_features=encoder_layers ,activation_function=activation_function)}
    
        # print()
 
        # Instantiate parameters of the model
        for model in models_ppo.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.05)
        
        self.cfg_ppo["experiment"]["write_interval"] = 100
        # Define agent
        agent = PPO(models=models_ppo,
                memory=memory,
                cfg=self.cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
        
        #agent.load("agent_88000.pt")
        # Configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 1000000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # start training
        trainer.train()

    def start_training_sweep(self,n_sweeps):
        self.start_simulation()
       # Define sweep config
        sweep_configuration = {
            'method': 'bayes',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'Reward / Total reward (mean)'},
            'parameters': 
            {
                'mini_batches': {'values': [4, 8]},
                'lr': {'max': 0.003, 'min': 0.00003}
                
            }
        }

        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='isaac-rover-2.0')
        wandb.agent(sweep_id, function=self.sweep, count=n_sweeps)
        # Start sweep job.


    def sweep(self):
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb_name = f"test-Anton_{time_str}"
        run = wandb.init(project='isaac-rover-2.0', sync_tensorboard=True,name=self.wandb_name,group=self.wandb_group, entity="aalborg-university")
        self.cfg_ppo["learning_rate"] = wandb.config.lr
        self.cfg_ppo["mini_batches"] = wandb.config.mini_batches
        self.train()
       # wandb.finish()

    def start_training(self):
        self.start_simulation()
        wandb.init(project='isaac-rover-2.0', sync_tensorboard=True,name=self.wandb_name,group=self.wandb_group, entity="aalborg-university")
        self.train()
        wandb.finish()


    def start_training_sequential(self):
        for i in range(3):
            print(i)
            self.train()
        pass
if __name__ == '__main__':
    # Get hyperparameter config
    trainer = TrainerSKRL()
   # trainer.start_training_sequential()
    #trainer.start_training_sweep(4)
    trainer.start_training()
    #parse_hydra_configs()

    