import os
import copy
import time
import torch
import psutil
import numpy as np
import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt
from options import args_parser
from plot import perform_testing
from load_dataset import load_dataset
from WAT import weighted_average_time
from divide_data import create_client_server
from global_weight import Global_threshold
import json

def save_update_w(update_w, epoch, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'update_w_client0_epoch_0.pt')
    torch.save(update_w, file_path)
    print("----------MODEL SAVED----------")

#args = args_parser()
def train(options):
    # print(type(options))
    options_dict = vars(options)
    # for key, value in options_dict.items():
    #   print(f"{key}: {value}")

    predefined_args = [f'--{key}={value}' for key, value in options_dict.items()]
    # print(predefined_args)
    args = args_parser(predefined_args)

    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []
    all_loss_train = []
    all_loss_test = []
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.type == 'bin':
      dataset_train, dataset_test, flow_features , attack_labels = load_dataset('bin')
    elif args.type == 'mult':
      dataset_train, dataset_test, flow_features , attack_labels = load_dataset('mult')
    clients, server = create_client_server(args, dataset_train)
    print(colored('========================================================================================', 'yellow'))
    print(colored(f"Start training | Mode of Algorithm is: {args.mode}", 'red'))
    print(colored(f"Device is: {args.device} | Load dataset...", 'red'))
    print(colored('========================================================================================', 'yellow'))
    print(colored(f"{args.type}-Classification | Number of clients: {args.num_users} | Number of Iterations: {args.epochs}", 'blue'))
    print(colored('========================================================================================', 'yellow'))
    Iteration_start = time.time()
    process = psutil.Process(os.getpid())
    algorithm = Global_threshold()

    save_directory = '/home/ubuntu/EaaS/EaaS/Codes_based_Gas_pipline/MLP_based/client_weights_flask/'
    for iter in range(args.epochs):
      if iter == 0:
            server.clients_update_w, server.clients_loss = [], []
            client_training_times =[0] * args.num_users
            #for idx in range(args.num_users):
            start_client_time = time.time()
            update_w, loss = clients[0].train(0)
            end_client_time = time.time()
            client_training_time = end_client_time - start_client_time
            client_training_times[0] = client_training_time

            save_update_w(update_w, iter, save_directory)


            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)
            w_glob, loss_glob = server.FedAvg()
            #for idx in range(args.num_users):

            clients[0].update(w_glob,0)
            weighted_avg_time = weighted_average_time(client_training_times)
            selected_clients = [0] if client_training_times[0] <= weighted_avg_time else []


            if args.mode == 'Paillier':
                server.model.load_state_dict(copy.deepcopy(clients.model.state_dict()))
            print(colored('========== Epoch {:3d} =========='.format(iter), 'yellow'))
            # testing
            all_acc_train, all_acc_test, all_loss_glob, all_loss_train, all_loss_test, acc_test = perform_testing(server,loss_glob, dataset_train, dataset_test,
                                                                                                        all_acc_train, all_acc_test, all_loss_glob,
                                                                                                        all_loss_train, all_loss_test)

            #com_cost1 = algorithm.communication(1, args.num_users)
      else:
            server.clients_update_w, server.clients_loss = [], []
            for idx in selected_clients:
                update_w, loss = clients[idx].train(idx)
                server.clients_update_w.append(update_w)
                server.clients_loss.append(loss)

            save_update_w(update_w, iter, save_directory)

            w_glob, loss_glob = server.FedAvg()

            clients[0].update(w_glob,0)

            if args.mode == 'Paillier':
                server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
            print(colored('========== Epoch {:3d} =========='.format(iter), 'yellow'))
            # testing
            all_acc_train, all_acc_test, all_loss_glob, all_loss_train, all_loss_test, acc_test = perform_testing(server,loss_glob, dataset_train, dataset_test,
                                                                                                        all_acc_train, all_acc_test, all_loss_glob,
                                                                                                        all_loss_train, all_loss_test)
    
    
    #com_cost2 = algorithm.communication(args.epochs - 1 , len(selected_clients))
    #final_com_cost = com_cost1 + com_cost2
    #final_com_cost_formatted = '{:.3f}MB'.format(final_com_cost)
    Iteration_end = time.time()
    memoryTraining=process.memory_percent()
    #torch.save(server.model,'/content/drive/MyDrive/EaaS/Samira_modellmultgas.pth')
    print(colored('========================================================================================', 'yellow'))
    #print('final_com_cost:', colored(final_com_cost_formatted, 'blue'))
    print('%s seconds Convergence Speed(Overall Training time):', Iteration_end - Iteration_start)
    print("---Memory---",memoryTraining)
    print(colored('========================================================================================', 'yellow'))


    return attack_labels, flow_features, acc_test
if __name__ == '__main__':
  train(args.type)