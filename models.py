import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torch.nn.functional as F

def create_model(args):
    state = None
    # Load a model or create a brand new one
    if args.pretrained:
        print "=> Loading pretrained model from " + args.pretrained  
        assert os.path.isfile(args.pretrained), "[!] Pretrained model " + args.pretrained + " doesn't exist"
        model = torch.load(args.pretrained)
        assert model != None, "[!] Failed to load " + args.pretrained
        # Start from epoch-1
        model.start_epoch = 1
    else:
        print "=> Creating a " + args.model
        model = globals()[args.model + 'Model'](args)
        model.start_epoch = 1

    if args.resume:
        print "=> Loading checkpoints from " + args.resume
        assert os.path.isfile(args.resume), "[!] Checkpoint " + args.resume + " doesn't exist" 
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    if args.nGPU > 0:
        cudnn.benchmark = True
    if args.nGPU > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    criterion = nn.__dict__[args.criterion + 'Loss']()
    if args.nGPU > 0:
        criterion = criterion.cuda()

    return model, criterion, state 

class LeNetModel(nn.Module):
    def __init__(self, args):
        super(LeNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
