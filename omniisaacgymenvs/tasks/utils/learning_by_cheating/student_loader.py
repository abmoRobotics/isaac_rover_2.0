from omniisaacgymenvs.tasks.utils.learning_by_cheating.student_model import Student
import torch

class student_loader():
    def __init__(self, num_envs, info, device='cuda:0', model_path="best.pt") -> None:
        self.cfg = self.cfg_fn()
        self.info = info
        self.model = self.load_model(model_path)
        self.h = self.model.belief_encoder.init_hidden(num_envs).to(device)
        

    def load_model(self, model_name):
        model = Student(self.info, self.cfg)
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.cuda()

        return model

    def act(self,observations):
        with torch.no_grad():
            #print(observations.shape)
            actions, predictions, self.h = self.model(observations.unsqueeze(1),self.h)
            return actions

    def info_fn(self):
        pass

    def cfg_fn(self):
        cfg = {
            "info":{
                "reset":            0,
                "actions":          0,
                "proprioceptive":   0,
                "exteroceptive":    0,
            },
            "learning":{
                "learning_rate": 1e-4,
                "epochs": 500,
                "batch_size": 8,
            },
            "encoder":{
                "activation_function": "leakyrelu",
                "encoder_features": [80,60]},

            "belief_encoder": {
                "hidden_dim":       300,
                "n_layers":         2,
                "activation_function":  "leakyrelu",
                "gb_features": [128,128,120],
                "ga_features": [128,128,120]},

            "belief_decoder": {
                "activation_function": "leakyrelu",
                "gate_features":    [128,256,512],
                "decoder_features": [128,256,512]
            },
            "mlp":{"activation_function": "leakyrelu",
                "network_features": [256,160,128]},
                }

        return cfg





