

from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
from model import Net, train, test

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

        
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy for _, val in self.model.state.dict().items()]
        

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
         
         
        #lokales Training
        train(self.model,  self.trainloader, optim, epochs, self.device)
        
        return self.get_parameters(), len(self.trainloader), {}
    
    
def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):

    self.set_parameters(parameters)
    loss, accuracy = test(self.model, self.valloader, self.device)
    return float(loss), len(self.valloader), {'accuracy': accuracy}
