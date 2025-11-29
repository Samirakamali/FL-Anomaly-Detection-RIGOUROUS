import os
import json
import time
import copy
import torch
import requests
from termcolor import colored
from options import args_parser
from flask_inference2 import prediction
#from flask_inference import prediction
from load_dataset2 import load_dataset
from divide_data import create_client_server
from plot import perform_testing
from global_weight import Global_threshold
from models.Nets import MLP
import stix2
from stix2 import CustomObject, properties, Bundle, Relationship, parse
from kafka_publishing import send_stix_files

@CustomObject('model', [
    ('model_id', properties.StringProperty(required=True)),
    ('model_name', properties.StringProperty(required=True)),
    ('learning_type', properties.StringProperty(required=True)),
    ('flow_features', properties.ListProperty(properties.StringProperty, required=True)),
    ('multi_class_accuracy', properties.FloatProperty(required=False)),
    ('comparison_metric', properties.StringProperty(required=True)),
    ('framework', properties.StringProperty(required=False)),
    ('framework_version', properties.StringProperty(required=False)),
    ('collection_tool', properties.StringProperty(required=False)),
    ('collection_tool_version', properties.StringProperty(required=False)),
    ('training_timestamp', properties.IntegerProperty(required=False)),
])
class Metadata(object):
    pass


def save_weight_and_loss(update_w, loss, training_time, client_id, epoch, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    weight_path = os.path.join(base_dir, f"update_w_{client_id}_epoch_{epoch}.pt")
    loss_path = os.path.join(base_dir, f"loss_{client_id}_epoch_{epoch}.json")
    torch.save(update_w, weight_path)
    with open(loss_path, 'w') as f:
        json.dump({'epoch': epoch, 'loss': loss, 'time': training_time, 'id': client_id}, f)
    return weight_path, loss_path

def send_to_server(weight_path, loss_path, server_url):
    with open(weight_path, 'rb') as wf, open(loss_path, 'r') as lf:
        json_payload = json.load(lf)
        files = {
            'json': ('json_data', json.dumps(json_payload), 'application/json'),
            'model': (os.path.basename(weight_path), wf, 'application/octet-stream')
        }
        requests.post(f"{server_url}/cmodel", files=files)

def wait_agg_model_loss(client_id, epoch, recv_dir):
    os.makedirs(recv_dir, exist_ok=True)
    model_path = os.path.join(recv_dir, f"w_glob_epoch_{epoch}.pth")
    loss_path = os.path.join(recv_dir, f"global_loss_epoch_{epoch}.json")

    while not (os.path.exists(model_path) and os.path.exists(loss_path)):
        print(f"[{client_id}] Waiting for aggregated model and loss for epoch {epoch}...")
        time.sleep(5)

    with open(loss_path, 'r') as f:
        loss_data = json.load(f)
        global_loss = loss_data['loss']
        acc = loss_data.get('acc', 0)
        precision = loss_data.get('precision', 0)
        f1 = loss_data.get('f1', 0)
        average_time = loss_data.get('average_time', 0)

    return  model_path, global_loss, acc, precision, f1, average_time

def get_model_info(model_instance, client_id):
    model_name = model_instance.__class__.__name__
    if model_name == 'MLP':
        model_id = f"{client_id}_model"
        learning_type = "supervised"
    elif model_name == 'CNN':
        model_id = f"{client_id}_model"
        learning_type = "supervised"
    return model_id, model_name, learning_type


all_acc_train =  []
all_acc_test =   []
all_loss_glob =  []
all_loss_train = []
all_loss_test =  []

def train(options, client_id, server_url="http://0.0.0.0:8803"):
    args = args_parser([f"--{k}={v}" for k, v in vars(options).items()])
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    dataset_train, dataset_test, flow_features, attack_labels = load_dataset(args.type)
    clients, server = create_client_server(args, dataset_train)

    model_instance = MLP(args=args)

    print(colored(f"[{client_id}] Federated Training Started...", "cyan"))

    final_acc = 0.0
    final_precision = 0.0
    final_f1 = 0.0

    for iter in range(args.epochs):
        if iter == 0:
            print(colored(f"[{client_id}] Epoch {iter} training...", "yellow"))
            start_time = time.time()
            update_w, loss = clients[0].train(0)
            end_time = time.time()
            training_time = end_time - start_time

            weight_dir = f"/home/ubuntu/API_integration_device/Client1/client_weights_flask"
            weight_path, loss_path = save_weight_and_loss(update_w, loss, training_time, client_id, iter, weight_dir)
            send_to_server(weight_path, loss_path, server_url)

            recv_dir = f"/home/ubuntu/API_integration_device/Client1/model_update"
            agg_model_path, global_loss, acc, precision, f1, average_time = wait_agg_model_loss(client_id, iter, recv_dir)

            new_state_dict = torch.load(agg_model_path, map_location='cpu')
            
            clients[0].update(new_state_dict, 0)

            if args.mode == 'Paillier':
                server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
            print(colored('========== Epoch {:3d} =========='.format(iter), 'yellow'))

            final_acc = acc
            final_precision = precision
            final_f1 = f1

            #perform_testing(server,global_loss, dataset_train, dataset_test,all_acc_train, all_acc_test, all_loss_glob, all_loss_train, all_loss_test)
        else:
            if training_time <= average_time:
                print(colored(f"[{client_id}] Epoch {iter} training...", "yellow"))         
                update_w, loss = clients[0].train(0)

                weight_dir = f"/home/ubuntu/API_integration_device/Client1/client_weights_flask"
                weight_path, loss_path = save_weight_and_loss(update_w, loss, training_time, client_id, iter, weight_dir)
                send_to_server(weight_path, loss_path, server_url)

                recv_dir = f"/home/ubuntu/API_integration_device/Client1/model_update"
                agg_model_path, global_loss, acc, precision, f1, average_time = wait_agg_model_loss(client_id, iter, recv_dir)

                new_state_dict = torch.load(agg_model_path, map_location='cpu')
                
                clients[0].update(new_state_dict, 0)

                if args.mode == 'Paillier':
                    server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
                print(colored('========== Epoch {:3d} =========='.format(iter), 'yellow'))

                final_acc = acc
                final_precision = precision
                final_f1 = f1

                #perform_testing(server,global_loss, dataset_train, dataset_test,all_acc_train, all_acc_test, all_loss_glob, all_loss_train, all_loss_test)
            else:
                print(f"[{client_id}] Skipping training in epoch {iter} due to high training time ({training_time:.2f} > {average_time:.2f}).")

    torch.save(server.model,'/home/ubuntu/API_integration_device/Client1/inference_results/modetrained_mult.pth') 
    print(colored(f"[{client_id}] Training finished.", "green"))

    model_id, model_name, learning_type = get_model_info(model_instance, client_id)

    stix_obj = Metadata(
        model_id=model_id,
        model_name=model_name,
        learning_type=args.type,
        flow_features=flow_features,
        multi_class_accuracy=final_acc,
        comparison_metric="softmax",
        framework="PyTorch",
        framework_version=torch.__version__,
        collection_tool="",
        collection_tool_version="",
        training_timestamp=int(time.time())
    )
    
    print(f"[{client_id}] Final STIX metrics -> acc: {final_acc}, precision: {final_precision}, f1: {final_f1}")

    validation_data_path =  '/home/ubuntu/API_integration_device/Client1/Data/device1_testsandbox.csv'      

    client_ip, client_port = "0.0.0.0", 8002 
    server_ip, server_port = "0.0.0.0", 8803   

    _, stix_anomalies_path = prediction(validation_data_path, client_ip, client_port, server_ip, server_port)

    bundle = parse(open(stix_anomalies_path).read(), allow_custom=True)
    anomaly_objects = [obj for obj in bundle.objects if obj.type == 'anomaly']

    relationships = [
        Relationship(relationship_type='indicates',
                     source_ref=stix_obj.id,
                     target_ref=anomaly.id)
        for anomaly in anomaly_objects
    ]

    bundle = Bundle(objects=[stix_obj] + anomaly_objects + relationships)

    output_path = '/home/ubuntu/API_integration_device/Client1/combined_output_bundle.stix'
    with open(output_path, 'w') as f:
        f.write(bundle.serialize(pretty=True))

    print(f"[{client_id}] Combined STIX bundle saved to: {output_path}")

    send_stix_files()


    return attack_labels, flow_features, bundle

if __name__ == '__main__':
    args = args_parser([])
    train(args)