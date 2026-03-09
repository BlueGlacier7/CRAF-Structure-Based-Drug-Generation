from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)
    def close(self):
        self.writer.close()
