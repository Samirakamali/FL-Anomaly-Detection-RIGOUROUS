from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, utils
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import torch

def load_dataset(type_class):
    dataset_path = '/home/ubuntu/EaaS/API_integration_server/Aggregator_server/Data/wustl_iiot_2021.csv'
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.drop(['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId'], axis=1)
    encoded_labels = {'normal': 4, 'DoS': 0, 'Reconn': 1, 'Backdoor': 2, 'CommInj': 3} 
    dataset['EnLabels'] = dataset['Traffic'].map(encoded_labels) #encoded_labels

    attack_labels = ["Command Injection", "DoS"," Reconnaissance","Backdoor"]
    flow_features = ["id","Tpkts","Sbytes","Dbytes","TBytes","Sload","Dload","Tload","Srate","Drate","Trate","Sloss","Dloss","Tloss",
                    "Ploss","ScrJitter","DrcJitter","cycletime","SIntPkt","DIntPkt","Proto","Dur","TcpRtt", "Idle","sum","min","time",
                    "max","sDSb","sTtl","dTtl","SAppBytes","DAppBytes","TotAppByte","SynAck","RunTime","sTos","SrcJitAct","DstJitAct"
                     ]
    xs = dataset.drop(['Traffic', 'Target','EnLabels'], axis=1)
    if type_class == 'bin':
      ys = dataset['Target'].values #Binary class classification
    elif type_class == 'mult':
      ys = dataset['EnLabels'].values #Multi_class classification

    x_train, x_temp, y_train, y_temp = train_test_split(xs, ys, test_size=0.2, random_state=25)
    x_test, x_validate, y_test, y_validate = train_test_split(x_temp, y_temp, test_size=0.2, random_state=200)
    min_max_scaler = MinMaxScaler().fit(x_train)

    x_train = min_max_scaler.transform(x_train)
    x_validate = min_max_scaler.transform(x_validate)
    x_test = min_max_scaler.transform(x_test)
    # print(f'-----x_train-----{x_train.shape},{y_train.shape}-----y_train-----')
    
    tx_train = torch.tensor(x_train,dtype=torch.float32)
    tx_test = torch.tensor(x_test,dtype=torch.float32)
    ty_train = torch.LongTensor(y_train)
    ty_test = torch.LongTensor(y_test)

    dataset_train = TensorDataset(tx_train,ty_train)
    dataset_test = TensorDataset(tx_test,ty_test)
                 
    return dataset_train, dataset_test, flow_features, attack_labels