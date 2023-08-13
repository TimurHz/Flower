

from collections import OrderedDict
import torch
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 validationloader,
                 num_classes) -> None:
        super().__init__()
        
        self.trainloader = trainloader
        self.valloader = validationloader
        
        self.model = Net(num_classes)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
    def set_parameters(self, parameters):
        
        params_dict = zip(self.model.state_dicr().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        lr = config['lr']
        momentum = config['momentum']
        
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
         
         
        #lokales Training
        
        