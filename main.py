from opts import args
from models import create_model
from datasets import create_loader
from log import Logger
from train import Trainer

def main():
    # Create Model, Criterion and State
    model, criterion, state = create_model(args)
    print "=> Model and criterion are ready"
    # DEBUG
    exit(0)
    #
    # Create Dataloader
    train_loader = create_loader(args, 'train')
    val_loader = create_loader(args, 'val')
    print "=> Dataloaders are ready"
    # Create logger
    logger = Logger(args, state)
    trainer = Trainer(args, model, criterion, logger)
    print "=> Trainer is ready"

    if args.test_only:
        test_summary = trainer.test(0, val_loader)
        print "Top1 Error: %6.3f  Top5 Error: %6.3f" % (test_summary['top1'], test_summary['top5'])
        # If unnecessary, comment the following line
        logger.record(0, test_summary=test_summary)
        return 0 

    start_epoch = logger.state['epoch'] + 1
    print "=> Start training"
    for epoch in xrange(start_epoch, args.n_epochs + 1):
        train_summary = trainer.train(epoch, train_loader)
        test_summary = trainer.test(epoch, val_loader)

        logger.record(epoch, train_summary=train_summary, test_summary=test_summary, model=model) 

    logger.final_print()

if __name__ == '__main__':
    main()
