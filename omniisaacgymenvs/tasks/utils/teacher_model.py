import torch
import torch.nn as nn
from torch.distributions import Normal

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
        self.layer = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(
            self, info, cfg, encoder=""):
        super(Encoder,self).__init__()
        encoder_features = cfg["encoder_features"]
        activation_function = cfg["activation_function"]
        
        self.encoder = nn.ModuleList() 
        in_channels = info[encoder]
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

    def forward(self, x):
        
        for layer in self.encoder:
            x = layer(x)
        
        return x

class MLP(nn.Module):
    def __init__(
            self, info, cfg, belief_dim):
        super(MLP,self).__init__()
        self.network = nn.ModuleList()  # MLP for network
        proprioceptive = info["proprioceptive"]
        action_space = info["actions"]
        activation_function = cfg["activation_function"]
        network_features = cfg["network_features"]

        in_channels = proprioceptive + belief_dim
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def forward(self, p, belief):
        
        x = torch.cat((p,belief),dim=2)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter


class Teacher(nn.Module):
    def __init__(
            self, info, cfg, teacher):
        super(Teacher,self).__init__()

        self.n_re = info["reset"]
        self.n_pr = info["proprioceptive"]
        self.n_sp = info["sparse"]
        self.n_de = info["dense"]
        self.n_ac = info["actions"]
        
        self.encoder1 = Encoder(info, cfg["encoder"], encoder="sparse")
        self.encoder2 = Encoder(info, cfg["encoder"], encoder="dense")
        encoder_dim = cfg["encoder"]["encoder_features"][-1] * 2
        self.MLP = MLP(info, cfg["mlp"], belief_dim=120)
        # Load teacher policy
        teacher_policy = torch.load(teacher)["policy"]
        # Filter out encoder to only maintain network MLP
        mlp_params = {k: v for k,v in teacher_policy.items() if ("network" in k or "log_std_parameter" in k)}
        #print(mlp_params.keys())
        encoder_params1 = {k[9:]: v for k,v in teacher_policy.items() if "encoder0" in k}
        encoder_params2 = {k[9:]: v for k,v in teacher_policy.items() if "encoder1" in k}
        # Load state dict
        self.MLP.load_state_dict(mlp_params)
        self.encoder1.load_state_dict(encoder_params1)
        self.encoder2.load_state_dict(encoder_params2)
        

    def forward(self, x):
        n_ac = 0#self.n_ac
        n_pr = self.n_pr
        n_re = self.n_re
        n_sp = self.n_sp
        n_de = self.n_de
        reset = x[:,:, 0:n_re]
        actions = x[:,:,n_re:n_re+n_ac]
        
        proprioceptive = x[:,:,n_re+n_ac:n_re+n_ac+n_pr]
        sparse = x[:,:,-(n_sp+n_de):-n_de]
        dense = x[:,:,-n_de:]
        exteroceptive = torch.cat((sparse,dense),dim=2)

        # n_p = self.n_p
        
        # p = x[:,:,0:n_p]        # Extract proprioceptive information  
        
        # e = x[:,:,n_p:1084]         # Extract exteroceptive information
        
        e_l1 = self.encoder1(sparse) # Pass exteroceptive information through encoder
        
        e_l2 = self.encoder2(dense)
        e_l = torch.cat((e_l1,e_l2), dim=2)
                
        actions, log_std = self.MLP(proprioceptive,e_l)

        # min_log_std= -20.0
        # max_log_std = 2.0
        # log_std = torch.clamp(log_std, 
        #                         min_log_std,
        #                         max_log_std)

        
        # g_log_std = log_std
        # # print(actions.shape[0])
        # # print(actions.shape[2])
        # _g_num_samples = actions.shape[0]

        # # # distribution
        # _g_distribution = Normal(actions, log_std.exp())
        # #print(_g_distribution.shape)
        # # # sample using the reparameterization trick
        # actions = _g_distribution.rsample()
        # actions = torch.clamp(actions,-1.0,1.0)
        #print((actions-action).mean())
        return actions

