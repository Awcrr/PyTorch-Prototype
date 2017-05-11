import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# General options
# parser.add_argument('-data', required=True, help='Path to data directory')
# parser.add_argument('-dataset')
# parser.add_argument('-')

# Training options 

# Model options
parser.add_argument('-pretrained', default=None, help='Path to the pretrained model')
parser.add_argument('-resume', default=None, help='Path to resume training from a previous checkpoint')
parser.add_argument('-model', default='LeNet', help='Model type when we create a new one')
parser.add_argument('-nGPU', default=1, type=int, help='Number of GPUs for training')
parser.add_argument('-criterion', default='CrossEntropy', help='Type of objective function')

args = parser.parse_args()
