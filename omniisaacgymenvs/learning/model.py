
from ast import Mod
from xml.parsers.expat import model
from skrl.models.torch import GaussianMixin, Model
from skrl.models.torch import DeterministicMixin
from skrl.utils.model_instantiators import deterministic_model, Shape
import torch.nn as nn
import torch
from gym.spaces import Box

class ObserverationInfo():
    """
    ObserverationInfo is a class that stores information about the number of proprioceptive, sparse, dense, and beneath
    observations in a given observation space.

    Attributes:
        num_proprioceptive: An integer representing the number of proprioceptive observations in the observation.
        num_sparse: An integer representing the number of sparse observations in the observation.
        num_dense: An integer representing the number of dense observations in the observation.
        num_beneath: An integer representing the number of beneath observations in the observation.

    Methods:
        get_num_proprioceptive: Returns the number of proprioceptive observations in the observation.
        get_num_sparse: Returns the number of sparse observations in the observation.
        get_num_dense: Returns the number of dense observations in the observation.
        get_num_beneath: Returns the number of beneath observations in the observation.
    """
    def __init__(self, num_proprioceptive, num_sparse, num_dense, num_beneath):
        self.num_proprioceptive = num_proprioceptive
        self.num_sparse = num_sparse
        self.num_dense = num_dense
        self.num_beneath = num_beneath

    def get_num_proprioceptive(self):
        return self.num_proprioceptive
    def get_num_sparse(self):
        return self.num_sparse
    def get_num_dense(self):
        return self.num_dense
    def get_num_beneath(self):
        return self.num_beneath


class NetworkInfo():
    """
    This class is used to store the network architecture and activation function.

    The network architecture is stored as a list of integers, where each integer
    represents the number of features in a layer. The first integer is the number
    of features in the input layer, and the last integer is the number of features
    in the output layer.
    """
    def __init__(self, network_0, encoder_0, encoder_1, encoder_2, activation_function):
        self.mlp_features = network_0
        self.sparse_encoder_features = encoder_0
        self.dense_encoder_features = encoder_1
        self.beneath_encoder_features = encoder_2
        self.activation_function = activation_function

    def get_mlp_features(self):
        return self.mlp_features
    def get_sparse_encoder_features(self):
        return self.sparse_encoder_features
    def get_dense_encoder_features(self):
        return self.dense_encoder_features
    def get_beneath_encoder_features(self):
        return self.beneath_encoder_features
    def get_activation_function(self):
        return self.activation_function
    def get_total_length(self):
        return self.sparse_encoder_features[-1] + self.dense_encoder_features[-1] + self.beneath_encoder_features[-1]


class Layer(nn.Module):
    """
    Layer class is a wrapper for a linear layer with an activation function.
    It is used to create a neural network.
    
    Parameters
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    activation_function : str
        The activation function to be used.
        The available options are:
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
    
    Attributes
    ----------
    layer : nn.Sequential
        A sequential container for the linear layer and the activation function.
    activation_functions : dict
        A dictionary of the available activation functions.
    
    Methods
    -------
    forward(x)
        Forward pass of the layer.
    """
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
    """
    Encoder class.
    
    Parameters
    ----------
    input_dim : int
        The dimension of the input data.
    encoder_features : list of ints
        The number of features for each layer in the encoder.
    activation_function : str
        The activation function to use.
    
    Notes
    -----
    The encoder is a list of layers.
    """
    def __init__(self, input_dim, encoder_features=[80,60], activation_function="relu") -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        in_channels = input_dim
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return(x)


class StochasticActorHeightmap(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, networkInfo: NetworkInfo, observartionInfo: ObserverationInfo, device='cuda:0', clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0, reduction="sum"):
        #super().__init__(observation_space, action_space, device, clip_actions)
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.num_proprioception = observartionInfo.get_num_proprioceptive()
        self.num_sparse = observartionInfo.get_num_sparse()
        self.num_dense = observartionInfo.get_num_dense()
        self.num_beneath = observartionInfo.get_num_beneath()
        self.mlp_features = networkInfo.get_mlp_features()
        self.sparse_features = networkInfo.get_sparse_encoder_features()
        self.dense_features = networkInfo.get_dense_encoder_features()
        self.beneath_features = networkInfo.get_beneath_encoder_features()
        activation_function = networkInfo.get_activation_function()
        
        self.encoder0 = Encoder(self.num_sparse, self.sparse_features, activation_function)
        self.encoder1 = Encoder(self.num_dense, self.dense_features, activation_function)
        #self.encoder2 = Encoder(self.num_beneath,self.beneath_features,self.activation_function)

        self.num_exteroceptive = self.num_proprioception+self.sparse_features+self.num_dense  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroceptive 
        self.network = nn.ModuleList()  # MLP for network

        # Create MLP
        in_channels = self.num_proprioception + self.num_sparse + self.num_dense + self.num_beneath
        for feature in self.mlp_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        sparse = states[:,self.num_proprioception:self.num_proprioception+self.sparse_features]
        dense = states[:,self.num_proprioception+self.sparse_features:self.num_proprioception+self.sparse_features+self.num_dense]
        
        x0 = self.encoder0(sparse)
        x1 = self.encoder1(dense)
        x = torch.cat((states[:,0:self.num_proprioception], x0), dim=1)
        x = torch.cat((x, x1), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter

class DeterministicHeightmap(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space,  networkInfo: NetworkInfo, observartionInfo: ObserverationInfo, device='cuda:0', clip_actions=False):
        #super().__init__(observation_space, action_space, device, clip_actions)
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.num_proprioception = observartionInfo.get_num_proprioceptive()
        self.num_sparse = observartionInfo.get_num_sparse()
        self.num_dense = observartionInfo.get_num_dense()
        self.num_beneath = observartionInfo.get_num_beneath()
        self.mlp_features = networkInfo.get_mlp_features()
        self.sparse_features = networkInfo.get_sparse_encoder_features()
        self.dense_features = networkInfo.get_dense_encoder_features()
        self.beneath_features = networkInfo.get_beneath_encoder_features()
        activation_function = networkInfo.get_activation_function()
        
        self.encoder0 = Encoder(self.num_sparse, self.sparse_features, activation_function)
        self.encoder1 = Encoder(self.num_dense, self.dense_features, activation_function)
        #self.encoder2 = Encoder(self.num_beneath,self.beneath_features,self.activation_function)

        self.num_exteroceptive = self.num_proprioception+self.sparse_features+self.num_dense  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroceptive 
        self.network = nn.ModuleList()  # MLP for network

        # Create MLP
        in_channels = self.num_proprioception + self.num_sparse + self.num_dense + self.num_beneath
        for feature in self.mlp_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions, role):
        sparse = states[:,self.num_proprioception:self.num_proprioception+self.sparse_features]
        dense = states[:,self.num_proprioception+self.sparse_features:self.num_proprioception+self.sparse_features+self.num_dense]
        
        x0 = self.encoder0(sparse)
        x1 = self.encoder1(dense)
        x = torch.cat((states[:,0:self.num_proprioception], x0), dim=1)
        x = torch.cat((x, x1), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter
