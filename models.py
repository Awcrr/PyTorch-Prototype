import os

import torch
import torch.nn.init
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
    else:
        print "=> Creating a " + args.model
        model = globals()[args.model + 'Model'](args)

    if args.nGPU > 0:
        cudnn.benchmark = True
        if args.nGPU > 1:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # If resume, load all the states to the current model
    if args.resume:
        print "=> Loading checkpoints from " + args.resume
        assert os.path.exists(args.resume), "[!] Checkpoint " + args.resume + " doesn't exist" 
        # Load the latest checkpoint from a directory
        if os.path.isdir(args.resume):
            latest = torch.load(os.path.join(args.resume, 'latest.pt'))['latest']
            args.resume = os.path.join(args.resume, 'model_%d.pt' % latest)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

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

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for unit in self.modules():
            if isinstance(unit, nn.Conv2d):
                nn.init.kaiming_uniform(unit.weight, mode='fan_out')    
            elif isinstance(unit, nn.BatchNorm2d):
                nn.init.constant(unit.weight, 1)
                nn.init.constant(unit.bias, 0)
            elif isinstance(unit, nn.Linear):
                nn.init.normal(unit.weight, mean=0, std=0.01)
                nn.init.constant(unit.bias, 0)

class VGG16Model(nn.Module):
    def __init__(self, args):
        super(VGG16Model, self).__init__()
        self.feature = self._construct_features(args)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, args.num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        predictions = self.classifier(x)

        return predictions

    def _construct_features(self, args):
        layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        iChannels = 3
        feature_layers = []
      
        for layer in layers:
            if layer == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                feature_layers += [
                nn.Conv2d(iChannels, layer, kernel_size=3, padding=1),
                nn.BatchNorm2d(layer),
                nn.ReLU(True)]
                iChannels = layer

        return nn.Sequential(*feature_layers)

    def _initialize_weights(self):
        for unit in self.modules():
            if isinstance(unit, nn.Conv2d):
                nn.init.kaiming_uniform(unit.weight, mode='fan_out')    
            elif isinstance(unit, nn.BatchNorm2d):
                nn.init.constant(unit.weight, 1)
                nn.init.constant(unit.bias, 0)
            elif isinstance(unit, nn.Linear):
                nn.init.normal(unit.weight, mean=0, std=0.01)
                nn.init.constant(unit.bias, 0)
