import os
import torch

class Logger:
    def __init__(self, args, state):
        if not state:
            # Initial state
            self.state = {
                    'epoch': 0,
                    'best_top1': 0,
                    'best_top5': 0,
                    'optim': None}
        else:
            self.state = state
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
           os.mkdir(self.save_path) 

    def record(self, epoch, train_summary=None, test_summary=None, model=None):
        assert train_summary != None or test_summary != None, "Need at least one summary"    

        self.state['epoch'] = epoch
         
        if train_summary:
            train_top1 = train_summary['top1']
            train_top5 = train_summary['top5']
            self.state['optim'] = train_summary['optim'] 
            torch.save({'latest': epoch}, os.path.join(self.save_path, 'latest.pt')) 

        if test_summary:
            test_top1 = test_summary['top1']
            test_top5 = test_summary['top5']
            is_best = test_top1 > self.state['best_top1']
            if is_best:
                self.state['best_top1'] = test_top1
                self.state['best_top5'] = test_top5
                torch.save({
                    'state': self.state,
                    'model': model.state_dict() if model else None
                    }, os.path.join(self.save_path, 'best_model.pt'))
            torch.save({
                "state": self.state,
                "model": model.state_dict() if model else None
                }, os.path.join(self.save_path, 'model_%d.pt' % epoch))

    def final_print(self):
        print "- Best Top1: %6.3f Best Top5: %6.3f" % (self.state['best_top1'], self.state['best_top5'])
