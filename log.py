
class Logger:
    def __init__(self, state):
        if not state:
            # Initial state
            self.state = {
                    "best_top1": 100,
                    "best_top5": 100,
                    "optim": None}
        else:
            self.state = state

    def record(self, epoch, train_summary=None, test_summary=None, model=None):
        assert train_summary != None or test_summary != None, "Need at least one summary"    

    def final_print(self):
        pass
