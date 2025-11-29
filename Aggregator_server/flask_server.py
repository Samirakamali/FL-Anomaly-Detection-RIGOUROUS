from flask import Flask, request
from server import *
from models.Nets import MLP
from options import args_parser
import os, json, torch, copy, requests
from collections import defaultdict
from plot import perform_testing
from WAT import weighted_average_time
from load_dataset2 import load_dataset

all_acc_train = []
all_acc_test = []
all_loss_glob = []
all_loss_train = []
all_loss_test = []
args = args_parser()
dataset_train, dataset_test, flow_features, attack_labels = load_dataset(args.type)

app = Flask(__name__)
received_models = defaultdict(dict)
client_times = defaultdict(dict)
selected_clients_per_epoch = {}

@app.route('/')
def home():
    return "FL Server Running"

@app.route('/start_federated', methods=['POST'])
def start_federated():
    data = request.json
    if not data:
        return "Missing JSON body with parameters", 400

    num_iterations = str(data.get('num_iterations', 2))
    local_epoch = str(data.get('local_epoch', 5))
    classification_type = data.get('classification_type', 'mult')

    params = {
        'num_iterations': num_iterations,
        'local_epoch': local_epoch,
        'classification_type': classification_type
    }

    clients_file = "/home/ubuntu/EaaS/API_integration_server/Aggregator_server/clients.txt"
    with open(clients_file, 'r') as f:
        clients = [line.strip() for line in f if line.strip()]

    for client_url in clients:
        try:
            print(f"[Server] Sending params to client: {client_url}")
            response = requests.post(f"{client_url}/set_params", data=params)
            print(f"[Server] Response from {client_url}: {response.status_code}")
        except Exception as e:
            print(f"[Server] Failed to start client {client_url}: {e}")

    return "Training initiated on all clients"

@app.route('/cmodel', methods=['POST'])
def receive_model():
    model_data = request.files['model'].read()
    json_data = json.loads(request.files['json'].read().decode('utf-8'))

    epoch = str(json_data['epoch'])
    client_id = json_data['id']
    loss_value = json_data['loss']
    training_time = json_data.get('time', 0)
    client_times[epoch][client_id] = training_time

    save_dir = f"/home/ubuntu/EaaS/API_integration_server/Aggregator_server/load_model_serverflask/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"{client_id}_weights.pt")
    loss_path = os.path.join(save_dir, f"{client_id}_loss.json")

    with open(weight_path, 'wb') as f:
        f.write(model_data)
    with open(loss_path, 'w') as f:
        json.dump({'loss': loss_value}, f)

    received_models[epoch][client_id] = weight_path
    print(f"Length of received_models is: {len(received_models[epoch])}")

    aggregate_models(epoch, args)
    return "Model received."

def aggregate_models(epoch, args):
    epoch_dir = f"/home/ubuntu/EaaS/API_integration_server/Aggregator_server/load_model_serverflask/epoch_{epoch}"
    agg_dir = f"/home/ubuntu/EaaS/API_integration_server/Aggregator_server/aggregated_weights"
    os.makedirs(agg_dir, exist_ok=True)

    response_times = list(client_times[epoch].values())
    average_time = weighted_average_time(response_times)

    if int(epoch) == 0:
        selected_clients = [cid for cid, t in client_times[epoch].items() if t <= average_time]
        selected_clients_per_epoch[epoch] = selected_clients
        print(f"[Server] Selected clients for epoch {epoch}: {selected_clients}")
        print(f"[Server] Weighted average training time for epoch {epoch}: {average_time}")

        model = MLP(args=args)
        w_init = copy.deepcopy(model.state_dict())
    else:
        prev_path = os.path.join(agg_dir, f"w_glob_epoch_{int(epoch)-1}.pth")
        w_init = torch.load(prev_path)

    server = Server(args=args, w=copy.deepcopy(w_init))

    for file in os.listdir(epoch_dir):
        if file.endswith("_weights.pt"):
            w = torch.load(os.path.join(epoch_dir, file), map_location='cpu')
            server.clients_update_w.append(w)
        elif file.endswith("_loss.json"):
            with open(os.path.join(epoch_dir, file)) as f:
                loss_val = json.load(f)['loss']
                server.clients_loss.append(loss_val)

    w_glob, global_loss = server.FedAvg()

    model_path = os.path.join(agg_dir, f"w_glob_epoch_{epoch}.pth")
    loss_path = os.path.join(agg_dir, f"global_loss_epoch_{epoch}.json")
    torch.save(w_glob, model_path)

    acc_tr, acc_ts, loss_g, loss_tr, loss_ts, acc_test, precision, f1 = perform_testing(
        server,
        global_loss,
        dataset_train,
        dataset_test,
        all_acc_train,
        all_acc_test,
        all_loss_glob,
        all_loss_train,
        all_loss_test
    )

    with open(loss_path, 'w') as f:
        json.dump({
            'loss': global_loss,
            'epoch': epoch,
            'acc': acc_test,
            'precision': precision,
            'f1': f1,
            'average_time': average_time
        }, f)

    send_agg_to_clients(epoch)

def send_agg_to_clients(epoch):
    clients_file = "/home/ubuntu/EaaS/API_integration_server/Aggregator_server/clients.txt"
    with open(clients_file, 'r') as f:
        clients = [line.strip() for line in f if line.strip()]

    for c in clients:
        with open(f"/home/ubuntu/EaaS/API_integration_server/Aggregator_server/aggregated_weights/w_glob_epoch_{epoch}.pth", 'rb') as fmodel, \
             open(f"/home/ubuntu/EaaS/API_integration_server/Aggregator_server/aggregated_weights/global_loss_epoch_{epoch}.json", 'r') as floss:
            files = {
                'json': ('json_data', floss.read(), 'application/json'),
                'model': (f"w_glob_epoch_{epoch}.pth", fmodel, 'application/octet-stream')
            }
            try:
                print(f"[Server] Sending model to {c}")
                requests.post(f"{c}/aggmodel", files=files)
            except Exception as e:
                print(f"Failed to send model to {c}: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8803, debug=False)
