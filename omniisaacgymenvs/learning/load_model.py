    #from utils.model import Policy, Value
from learning.model import StochasticActor, StochasticCritic,StochasticActorHeightmap
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from typing import Union
from gym.spaces import Box
import gym

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



def load_model(model_name, features=[512,256,128]):
    observation_space = Box(-torch.inf,torch.inf,(3,))
    action_space = Box(-1.0,1.0,(2,))
    model = StochasticActorHeightmap(observation_space=observation_space, action_space=action_space, network_features=features)
    checkpoint = torch.load(model_name)
   # model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()
    return model

def load_value(model_name, features=[512,256,128]):
    observation_space = Box(-torch.inf,torch.inf,(3,))
    action_space = Box(-1.0,1.0,(2,))
    model = StochasticCritic(observation_space=observation_space, action_space=action_space, features=features)
    checkpoint = torch.load(model_name)
   # model.load_state_dict(checkpoint['state_dict'])
   # model.eval()
    #model.cuda()
    return model


model = load_model('./runs/ppoActor[512, 256, 128]relu_Critic[128, 64]relu_Encoder[80, 60]relu_test1_step100000/checkpoints/97000_policy.pt', features=[512,256,128])
value = load_value('./runs/ppoActor[512, 256, 128]relu_Critic[128, 64]relu_Encoder[80, 60]relu_test1_step100000/checkpoints/97000_policy.pt', features=[128,64])
a = torch.tensor([ [512.0,512.0,512.0]])

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
test = {"policy" : model,
        "value" : value}
b = [3]
observation_space = Box(-torch.inf,torch.inf,(3,))
action_space = Box(-1.0,1.0,(2,))
print(a)
print(action_space)
print(observation_space.shape[0])
agent = PPO(models=test,
            memory=None, 
            cfg=cfg_ppo, 
            observation_space=observation_space, 
            action_space=action_space,
            device='cuda:0')

#observation_space[2]
print(agent.policy.act(a,inference=True))
#print(agent.act(a, 0.0, 1000000000.0, inference=False))
#a = torch.tensor([4,4,4])
#print(model)
#print(model([1, 2, 3]))
#torch.load("./runs/22-03-29_06-59-09-455859_PPO/checkpoints/3000_policy.pt")