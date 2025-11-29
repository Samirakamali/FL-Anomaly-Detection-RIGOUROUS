
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn.functional import softmax
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import json
import stix2
from stix2 import CustomObject, properties, Bundle

@CustomObject('anomaly', [
    ('flow_id', properties.StringProperty(required=True)),
    ('value', properties.FloatProperty(required=True)),
    ('value_type', properties.StringProperty(required=True)),
    ('anomaly_name', properties.StringProperty(required=True)),
    ('source_ip', properties.StringProperty(required=True)),
    ('source_port', properties.IntegerProperty(required=True)),
    ('destination_ip', properties.StringProperty(required=True)),
    ('destination_port', properties.IntegerProperty(required=True)),
    ('flow_data', properties.ListProperty(properties.FloatProperty, required=True)),
    ('label', properties.StringProperty(required=True)),
])
class Anomaly(object):
    pass

def prediction(dataset_path, client_ip, client_port, server_ip, server_port):
    dataset1 = pd.read_csv(dataset_path)
    encoded_labels = {'BENIGN': 0, 'DoS Hulk': 1, 'DoS slowloris': 2, 'GoldenEye': 3, 'Slowloris': 4, 'Hulken': 5}
    dataset1['muntiLabel'] = dataset1['Label'].map(encoded_labels)
    dataset1['binLabel'] = np.where(dataset1['muntiLabel'] == 0, 0, 1)

    X_validate = dataset1.drop(['Label', 'muntiLabel', 'binLabel'], axis=1)
    Y_validate = dataset1['muntiLabel'].values

    x_validate = MinMaxScaler().fit_transform(X_validate)

    tx_test = torch.tensor(x_validate, dtype=torch.float32)
    ty_test = torch.LongTensor(Y_validate)
    dataset_test = TensorDataset(tx_test, ty_test)

    model_path = '/home/ubuntu/API_integration_device/Client0/inference_results/modetrained_mult.pth'
    mymodel = torch.load(model_path)
    model_name = str(type(mymodel)).split("'")[1].split(".")[-1]
    model_id = "model1" if model_name == 'MLP' else "unknown"

    data_loader = DataLoader(dataset_test, batch_size=64)
    all_predictions = []

    for idx, (data, target) in enumerate(data_loader):
        preds = mymodel(data)
        pred_probs = softmax(preds, dim=1).detach().numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
        anomaly_names = ['BENIGN', 'DoS Hulk', 'DoS slowloris', 'GoldenEye', 'Slowloris', 'Hulken']
        anomaly_name = [anomaly_names[label] for label in pred_labels]

        for i in range(len(data)):
            flow_data = data[i].tolist()
            row = {
                'flow_id': idx * len(data) + i,
                'flow_data': flow_data,
                'anomaly_label': pred_labels[i],
                'anomaly_name': anomaly_name[i],
                'anomaly_score': pred_probs[i].max()
            }
            all_predictions.append(row)

    predictions_df = pd.DataFrame(all_predictions)
    df_filtered = predictions_df[predictions_df['anomaly_name'] != 'BENIGN'].head(1)

    json_data = []
    for _, row in df_filtered.iterrows():
        flow_info = {
            "flow_id": row["flow_id"],
            "anomaly_score": row["anomaly_score"],
            "anomaly_label": row["anomaly_label"],
            "anomaly_name": row["anomaly_name"],
            "flow_data": row["flow_data"]
        }
        json_data.append(flow_info)

    output_json_path = "/home/ubuntu/API_integration_device/Client0/inference_results/inference_output.json"
    with open(output_json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    desired_json = {
        "flow_filename": "/home/ubuntu/API_integration_device/Client0/Data/validation_data.csv",
        "model_id": model_id,
        "anomaly_results": [json_data]
    }
    with open("/home/ubuntu/API_integration_device/Client0/inference_results/inference_output.json.json", "w") as json_out:
        json.dump(desired_json, json_out, indent=4)

    stix_objects = []
    for _, row in df_filtered.iterrows():
        anomaly = Anomaly(
        flow_id=str(row["flow_id"]),
        value=float(row["anomaly_score"]),
        value_type="confidence",
        anomaly_name=row["anomaly_name"],
        source_ip=client_ip,
        source_port=client_port,
        destination_ip=server_ip,
        destination_port=server_port,
        flow_data=row["flow_data"],
        label=row["anomaly_name"].lower()
        )
        stix_objects.append(anomaly)

    STIX_file = "/home/ubuntu/API_integration_device/Client0/Inference_data.stix"
    bundle = Bundle(objects=stix_objects)
    with open(STIX_file, "w") as f:
        f.write(bundle.serialize(pretty=True))

    return desired_json, STIX_file
