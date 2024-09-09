import torch
import torch.nn as nn

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

class Belief_Encoder(nn.Module):
    def __init__(
            self, info, cfg, input_dim=120):
        super(Belief_Encoder,self).__init__()
        self.hidden_dim = cfg["hidden_dim"]
        self.n_layers = cfg["n_layers"]
        activation_function = cfg["activation_function"]
        proprioceptive = info["proprioceptive"]
        input_dim = proprioceptive+input_dim
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.gb = nn.ModuleList()
        self.ga = nn.ModuleList()
        gb_features = cfg["gb_features"]
        ga_features = cfg["ga_features"]

        in_channels = self.hidden_dim
        for feature in gb_features:
            self.gb.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        in_channels = self.hidden_dim
        for feature in ga_features:
            self.ga.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.ga.append(nn.Sigmoid())

    def forward(self, p, l_e, h):
        # p = proprioceptive
        # e = exteroceptive
        # h = hidden state
        # x = input data, h = hidden state
        x = torch.cat((p,l_e),dim=2)
        out, h = self.gru(x, h)
        x_b = x_a = out
        
        for layer in self.gb:
            x_b = layer(x_b)
        for layer in self.ga:
            x_a = layer(x_a)
        x_a = l_e * x_a
        # TODO IMPLEMENT GATE
        belief = x_b + x_a

        return belief, h, out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        return hidden

class Belief_Decoder(nn.Module):
    def __init__(
            self, info, cfg, n_input=50, hidden_dim=50,n_layers=2,activation_function="leakyrelu"):
        super(Belief_Decoder,self).__init__()
        exteroceptive = info["sparse"] + info["dense"]
        gate_features = cfg["gate_features"] #[128,256,512, exteroceptive]
        decoder_features = cfg["decoder_features"]#[128,256,512, exteroceptive]
        #n_input = cfg[""]
        gate_features.append(exteroceptive)
        decoder_features.append(exteroceptive)
        self.n_input = n_input
        self.gate_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
    

        in_channels = self.n_input
        for feature in gate_features:
            self.gate_encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        self.gate_encoder.append(nn.Sigmoid())  

        in_channels = self.n_input
        for feature in decoder_features:
            self.decoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        

    def forward(self, e, h):
        gate = h[-1]
        decoded = h[-1]
        for layer in self.gate_encoder:
            gate = layer(gate)

        for layer in self.decoder:
            decoded = layer(decoded)
        x = e*gate
        x = x + decoded
        return x
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(1.0)

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

class Student(nn.Module):
    def __init__(
            self, info, cfg):
        super(Student,self).__init__()

        self.n_re = info["reset"]
        self.n_pr = info["proprioceptive"]
        self.n_sp = info["sparse"]
        self.n_de = info["dense"]
        self.n_ac = info["actions"]
        
        self.encoder1 = Encoder(info, cfg["encoder"], encoder="sparse")
        self.encoder2 = Encoder(info, cfg["encoder"], encoder="dense")
        encoder_dim = cfg["encoder"]["encoder_features"][-1] * 2
        self.belief_encoder = Belief_Encoder(info, cfg["belief_encoder"], input_dim=encoder_dim)
        self.belief_decoder = Belief_Decoder(info, cfg["belief_decoder"], cfg["belief_encoder"]["hidden_dim"])
        self.MLP = MLP(info, cfg["mlp"], belief_dim=120)
        # Load teacher policy
        teacher_policy = torch.load("agent_219000.pt")["policy"]
        #Filter out encoder to only maintain network MLP
        # print(teacher_policy.keys())

        # mlp_params = {k: v for k,v in teacher_policy.items() if ("network" in k or "log_std_parameter" in k)}
        # encoder_params1 = {k[9:]: v for k,v in teacher_policy.items() if "encoder0" in k}
        # encoder_params2 = {k[9:]: v for k,v in teacher_policy.items() if "encoder1" in k}
        # # print(mlp_params.keys())
        # # print(encoder_params1.keys())
        # # print(encoder_params2.keys())
        # # Load state dict
        # self.MLP.load_state_dict(mlp_params)
        # self.encoder1.load_state_dict(encoder_params1)
        # self.encoder2.load_state_dict(encoder_params2)
        

    def forward(self, x, h):
        n_ac = 0#self.n_ac
        n_pr = self.n_pr
        n_re = 0#self.n_re
        n_sp = self.n_sp
        n_de = self.n_de
#        reset = x[:,:, 0:n_re]
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
        
        belief, h, out = self.belief_encoder(proprioceptive,e_l,h) # extract belief state
        
        #estimated = self.belief_decoder(exteroceptive,h)
        estimated = self.belief_decoder(exteroceptive,out)
        
        
        actions, log_std = self.MLP(proprioceptive,belief)

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
        #print((actions-action).mean())
        return actions, estimated, h
