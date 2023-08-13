import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    
    def __init__(self, num_classes: int) -> None:
        super(Net,self).__init__()