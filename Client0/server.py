import torch
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import MLP
from sklearn.metrics import precision_score, f1_score
import numpy as np
import pandas as pd
from options import args_parser

class Server():
    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = MLP(args=args).to(args.device)
        self.model.load_state_dict(w)
        # DP hyperparameters
        self.C = self.args.C
        self.sigma = self.args.sigma

    def FedAvg(self):
        if self.args.mode == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]   
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

        elif self.args.mode == 'DP':  # DP mechanism
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                # add gauss noise
                update_w_avg[k] += torch.normal(0, self.sigma**2 * self.C**2, update_w_avg[k].shape).to(self.args.device)
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)
        
        # Local paillier Encryption Aggregation
        elif self.args.mode == 'Paillier':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            # print(f'====update_w_avg====={update_w_avg.shape}, {type(update_w_avg)}===========')
            
            for k in update_w_avg.keys():
                client_num = len(self.clients_update_w)
                for i in range(1, client_num):  # client-wise sum
                    for j in range(len(update_w_avg[k])):  # element-wise sum
                        update_w_avg[k][j] += self.clients_update_w[i][k][j]
                        # print(f'=====update_w_avg[k][j]===={update_w_avg[k][j].shape}, {type(update_w_avg[k][j])}===========')
                for j in range(len(update_w_avg[k])):  # element-wise avg
                    update_w_avg[k][j] /= client_num
                
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)

    def test(self, datatest):
        self.model.eval()
        test_loss = 0
        correct = 0
        all_y_true = []
        all_y_pred = []
        args = args_parser()
          
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            # if self.args.gpu != -1:
            #     data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1].squeeze().cpu().numpy()
            y_true = target.data.cpu().numpy()
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            correct += (y_pred == y_true).sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        y_true_one_hot = np.eye(args.num_classes)[all_y_true]
        y_pred_one_hot = np.eye(args.num_classes)[all_y_pred]
        precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=1) * 100
        f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=1) * 100      

        return accuracy, test_loss, precision, f1
        
