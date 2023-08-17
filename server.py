from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from model import Net, test
import torch 


def get_on_fit_config(cfg: DictConfig):
    def fit_config_fn(server_round: int):
        
        return {'lr': cfg.lr, 'momentum': cfg.momentum, 'local_epochs': cfg.local_epochs}

    return fit_config_fn

def get_eval_fn(num_classes: int, testloader): 
     
    def evaluate_fn(server_round: int, params, cfg):
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dicr().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn