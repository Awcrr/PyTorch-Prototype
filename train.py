import time
import torch.optim as optim
from torch.autograd import Variable

class Trainer:
    def __init__(self, args, model, criterion, logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.SGD(
                    model.parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        self.nGPU = args.nGPU
        self.lr = args.lr

        if logger.state['optim']:
            self.optimizer.load_state_dict(logger.state['optim'])

    def train(self, epoch, train_loader):
        batch_time_avg = 0
        data_time_avg = 0
        loss_avg = 0
        top1_avg = 0
        top5_avg = 0
        total = 0
        n_batches = len(train_loader)
        
        model = self.model
        model.train()
        self.learning_rate(epoch)
        time_check_point = time.time()

        for i, (input, target) in enumerate(train_loader):
            data_time = time.time() - time_check_point   

            if self.nGPU > 0:
                # input = input.cuda()
                target = target.cuda(async=True)
            batch_size = target.size(0)
            input_var = Variable(input)
            target_var = Variable(target)

            output = model(input_var)
            loss = self.criterion(output, target_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            top1, top5 = self.accuracy(output.data, target, topk = (1, 5))

            batch_time = time.time() - time_check_point

            loss_avg += loss.data[0] * batch_size
            top1_avg += top1 * batch_size
            top5_avg += top5 * batch_size
            data_time_avg += data_time
            batch_time_avg += batch_time
            total += batch_size
            
            print "| Epoch[%d] [%d/%d]  Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f" % (
                    epoch,
                    i,
                    n_batches,
                    batch_time,
                    data_time,
                    loss.data[0],
                    top1,
                    top5)

            time_check_point = time.time()

        batch_time_avg /= n_batches 
        data_time_avg /= n_batches
        loss_avg /= total
        top1_avg /= total
        top5_avg /= total

        print "=> Epoch[%d]  Time_avg: %.3f  Data_avg: %.3f  Err_avg: %1.4f  top1: %7.3f  top5: %7.3f" % (
                    epoch,
                    batch_time_avg,
                    data_time_avg,
                    loss_avg,
                    top1_avg,
                    top5_avg) 

        return {'top1': top1_avg, 'top5': top5_avg, 'optim': self.optimizer.state_dict()}
        
    def test(self, epoch, test_loader):
        batch_time_avg = 0
        data_time_avg = 0
        top1_avg = 0
        top5_avg = 0
        total = 0
        n_batches = len(train_loader)

        model = self.model
        model.eval()
        time_check_point = time.time() 

        for i, (input, target) in enumerate(train_loader):
            data_time = time.time() - time_check_point   

            if self.nGPU > 0:
                input = input.cuda()
                target = target.cuda(async=True)
            batch_size = target.size(0)
            input_var = Variable(input)
            target_var = Variable(target)

            output = model(input_var)
            top1, top5 = self.accuracy(output.data, target, topk = (1, 5))

            batch_time = time.time() - time_check_point

            top1_avg += top1 * batch_size
            top5_avg += top5 * batch_size
            data_time_avg += data_time
            batch_time_avg += batch_time
            total += batch_size

            print "| Test[%d] [%d/%d]  Time %.3f  Data %.3f  top1 %7.3f  top5 %7.3f" % (
                    epoch,
                    i,
                    n_batches,
                    batch_time,
                    data_time,
                    top1,
                    top5)

        batch_time_avg /= n_batches
        data_time_avg /= n_batches
        top1_avg /= total
        top5_avg /= total

        print "=> Finished epoch[%d]  Top1 %7.3f  Top5 %7.3f\n" % (
                epoch,
                top1_avg,
                top5_avg)

        return {'top1': top1_avg, 'top5': top5_avg}

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100. / batch_size)[0])

        return res

    def learning_rate(self, epoch):
        lr = self.lr * (0.1 ** ((epoch - 1) // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
