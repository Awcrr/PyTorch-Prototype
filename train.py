import torch.optim as optim

class Trainer:
    def __init__(self, args, model, criterion):
        self.model = model
        self.criterion = criterion
         
