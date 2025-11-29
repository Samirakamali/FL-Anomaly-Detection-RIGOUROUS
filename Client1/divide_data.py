import copy
import numpy as np
from client import *
from server import *
from models.Nets import MLP

def create_client_server(args,dataset_train,):
    num_items = int(len(dataset_train) / args.num_users)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]
    net_glob = MLP(args=args).to(args.device)
    # divide training data - init models with same parameters 
    for i in range(args.num_users):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - new_idxs)
        new_client = Client(args=args, dataset=dataset_train , idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)
    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()))

    return clients, server