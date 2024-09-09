from omniisaacgymenvs.tasks.utils.learning_by_cheating.teacher_model import Teacher
import torch


class teacher_loader():
    def __init__(self,info,model_path="../../../agent_219000.pt") -> None:
        self.cfg = self.cfg_fn()
        self.info = info
        self.model = self.load_model(model_path)
        #self.h = self.model.belief_encoder.init_hidden(1).to('cuda:0')

    def load_model(self, model_path):
        model = Teacher(self.info, self.cfg, model_path)
        model.eval()
        model.cuda()

        return model

    def act(self,observations):
        with torch.no_grad():
            #print(observations.shape)
            actions= self.model(observations.unsqueeze(1))
            return actions

    def cfg_fn(self):
        cfg = {
            "info":{
                "reset":            0,
                "actions":          0,
                "proprioceptive":   0,
                "exteroceptive":    0,
            },
            "encoder":{
                "activation_function": "leakyrelu",
                "encoder_features": [80,60]},

            "mlp":{"activation_function": "leakyrelu",
                "network_features": [256,160,128]},
                }

        return cfg   

info = {
    "reset": 0,
    "actions": 2,
    "proprioceptive": 4,
    "sparse": 634,
    "dense": 1112}