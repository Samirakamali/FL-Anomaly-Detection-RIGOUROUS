from V1_IDEA_MLP import train, args_parser
from flask import Flask, request, send_file, jsonify
import os
import json
from stix2 import properties

app = Flask(__name__)
options = None


@app.route('/')
def hello():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Privacy-preserving FL-based Anomaly Detector</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container">
            <h1 class="mt-5">Privacy-preserving FL-based Anomaly Detector | Device1</h1>
            <form action="/set_params" method="post">
                <div class="form-group">
                    <label for="num_iterations">Number of Iterations</label>
                    <input type="number" class="form-control" id="num_iterations" name="num_iterations" required>
                </div>
                <div class="form-group">
                    <label for="local_epoch">Local Epoch</label>
                    <input type="number" class="form-control" id="local_epoch" name="local_epoch" required>
                </div>
                <div class="form-group">
                    <label for="classification_type">Type of Classification</label>
                    <select class="form-control" id="classification_type" name="classification_type">
                        <option value="bin">Binary</option>
                        <option value="mult">Multiclass</option>
                    </select>
                </div>
                <button type="submit" class="btn btn-primary">Set Parameters</button>
            </form>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    """

@app.route('/set_params', methods=['POST'])
def set_params():
    global options
    options = args_parser([])

    num_iterations = request.form['num_iterations']
    local_epoch = request.form['local_epoch']
    classification_type = request.form['classification_type']

    predefined_args = [

        '--epochs', num_iterations,
        '--local_ep', local_epoch,
        '--type', classification_type,

    ]
    options = args_parser(predefined_args)
    return model_train()

#@app.route('/start_train', methods=['POST'])
def model_train():
    global options
    if options is None:
        return "Please set parameters first!"
    attack_labels, flow_features, stix_obj = train(options, client_id="client1", server_url="http://0.0.0.0:8803")
    output_path = '/home/ubuntu/API_integration_device/Client1/metadata_model.stix'
    with open(output_path, 'w') as f:
        json.dump(stix_obj.serialize(), f, indent=4)

    return '''
        <html>
            <head><title>STIX Result</title></head>
            <body>
                <h2>Training complete.</h2>
                <a href="/download_stix" class="btn btn-primary">Download STIX Result (.stix)</a>
            </body>
        </html>
    '''

@app.route('/download_stix')
def download_stix():
    path = '/home/ubuntu/API_integration_device/Client1/combined_output_bundle.stix'
    return send_file(path, mimetype='application/json', as_attachment=True)


@app.route('/aggmodel', methods=['POST'])
def receive_agg_model():
    model_data = request.files['model'].read()
    loss_data = request.files['json'].read()
    
    # Parse JSON and get epoch number
    loss_json = json.loads(loss_data.decode('utf-8'))
    epoch = str(loss_json['epoch'])

    save_dir = "/home/ubuntu/API_integration_device/Client1/model_update"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"w_glob_epoch_{epoch}.pth")
    loss_path = os.path.join(save_dir, f"global_loss_epoch_{epoch}.json")

    # Save model
    with open(model_path, 'wb') as f:
        f.write(model_data)

    # Save loss
    with open(loss_path, 'w') as f:
        json.dump(loss_json, f)

    return f"Model and loss received for epoch {epoch}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=False)