from gym import spaces
import numpy as np
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.models.torch import GaussianMixin, Model
from skrl.models.torch import DeterministicMixin
import torch.nn as nn
import torch


class Layer(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Layer,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
           } 
        self.conv = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.conv(x)

class StochasticActorHeightmap(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, num_exteroception=2, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0, reduction="sum"):
        #super().__init__(observation_space, action_space, device, clip_actions)
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        x = states[:,self.num_proprioception:]
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat((states[:,0:self.num_proprioception], x), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter

class ppo_agent:
    def __init__(self, num_actions = 2,num_observations = 1088, policy_name="agent_280000.pt") -> None:
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.policy_name = policy_name
        self._setup()

    def _setup(self):
        num_actions = self.num_actions
        num_observations = self.num_observations

        action_space = spaces.Box(np.ones(num_actions) * -1.0, np.ones(num_actions) * 1.0)
        observation_space = spaces.Box(np.ones(num_observations) * -np.Inf, np.ones(num_observations) * np.Inf)



        models_ppo = {  "policy": StochasticActorHeightmap(observation_space, action_space, network_features=[256,160,128], encoder_features=[60,20], activation_function="leakyrelu"),
                    "value": None}


        self.agent = PPO(models=models_ppo,  # models dict
                    memory=None,  # memory instance, or None if not required
                    cfg=PPO_DEFAULT_CONFIG.copy(),  # configuration dict (preprocessors, learning rate schedulers, etc.)
                    observation_space=action_space,
                    action_space=observation_space,
                    device='cuda:0')

        self.agent.load(self.policy_name)

    def act(self,x):
        print(self.agent.policy.network)
        print(self.agent.policy)
        return self.agent.policy.act(x)[0]

# a = torch.ones(2,1088)*0.5
# agent = ppo_agent()
# agent.act(a)

a = torch.load("agent_280000.pt")["policy"]

keys = [k for k in a.keys() if "network" in k]
print(keys)



b = torch.load("best.pt")["state_dict"]

keys2 = [k for k in b.keys() if "MLP.network" in k]
b[keys2[0]] = a[keys[0]]
print(keys2)

c = {k[4:]: v for k,v in b.items() if "MLP.network" in k}
print(c)
print(c.keys())
#print(a["network"])

#    def ppo_agent(num_actions = 2,num_observations = 1088):
        



#    print(agent.policy.act(a)[1])
