from models.Nets import MLP
import numpy as np
from options import args_parser
args = args_parser()
class Global_threshold():
    def __init__(self):
        self.training_times = {}
        self.latency_times = {}
        self.global_threshold = None

    def calculate_weighted_average_time(self, times):
        weights = np.array([i * (i + 1) / 2 for i in range(1, len(times) + 1)])
        weighted_average = np.average(times, weights=weights)
        return weighted_average

    def update_global_threshold(self, list_time, list_delay):
        self.global_threshold = (list_time + list_delay) / 2
        return self.global_threshold

    def adaptive_threshold(self, global_threshold, sliding_window):
            alpha = 1
            adaptive_thresh = global_threshold + alpha
            return adaptive_thresh

    def communication(self, iteration, clients):
      model = MLP(args=args)
      param_size = 0
      for param in model.parameters():
        param_size += param.nelement() * param.element_size()
      modelsize = (param_size) / 1024**2
      print('Model size: {:.3f}MB'.format(modelsize))
      commcost = iteration * clients * modelsize
      return commcost